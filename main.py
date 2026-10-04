# EcoSort-AI classifier API
import io
import os
import secrets
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from PIL import Image, ImageOps

load_dotenv()

# config
API_KEY_NAME = "THE-API-KEY"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEY = os.getenv("API_KEY", "default-dev-key")

MODEL_FILENAME = os.getenv("MODEL_FILENAME", "ecosort-ai-sota.onnx")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

# transform configs
RESIZE_SIZE = 242
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ECOSORT_CLASSES = ["dry", "ewaste", "recyclable", "wet"]

model_session: ort.InferenceSession | None = None
input_name: str | None = None


# authorization
def get_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> str:
    if api_key is None or not secrets.compare_digest(api_key, VALID_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key !!",
        )
    return api_key


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_session, input_name

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at: {model_path}")

    model_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = model_session.get_inputs()[0].name

    out_shape = model_session.get_outputs()[0].shape
    n_out = out_shape[-1] if isinstance(out_shape[-1], int) else None
    if n_out is not None and n_out != len(ECOSORT_CLASSES):
        raise RuntimeError(
            f"Model outputs {n_out} classes but ECOSORT_CLASSES has {len(ECOSORT_CLASSES)}"
        )
    yield
    model_session = None


app = FastAPI(title="ECOSORT AI Classifier API", lifespan=lifespan)

allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # credentials can't be combined with a wildcard origin
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# preprocess custom image
def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Same as torchvision.transforms.Resize(size) on a PIL image:
    shorter edge -> size, aspect ratio preserved, bilinear."""
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Same rounding as torchvision.transforms.CenterCrop. arr is HWC."""
    h, w = arr.shape[:2]
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return arr[top:top + size, left:left + size, :]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """bytes -> float32 array of shape (1, 3, 224, 224), ImageNet-normalized."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
        img = img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not read image: {e}")

    img = _resize_shorter_side(img, RESIZE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC, [0,1]
    arr = _center_crop(arr, CROP_SIZE)
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))                   # CHW
    arr = np.expand_dims(arr, axis=0)                    # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1, keepdims=True)


# routing
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_session is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type !!")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file !!")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )

    try:
        input_tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    outputs = model_session.run(None, {input_name: input_tensor})
    logits = outputs[0]
    probabilities = softmax(logits)[0]

    k = min(3, len(ECOSORT_CLASSES))
    top_indices = np.argsort(probabilities)[::-1][:k]
    results = [
        {"class": ECOSORT_CLASSES[i], "confidence": round(float(probabilities[i]), 4)}
        for i in top_indices
    ]

    return {
        "filename": file.filename,
        "top_prediction": results[0]["class"],
        "confidence": results[0]["confidence"],
        "top_3_results": results,
    }