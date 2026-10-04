# base image
FROM python:3.13-slim
# working directory
WORKDIR /app
# copy  requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# copy rest of the files
COPY . .
# expose port 8000
EXPOSE 8000
# run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0","--port", "8000"]