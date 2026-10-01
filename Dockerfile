FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY analyzer/ analyzer/

CMD ["python", "-m", "analyzer", "run", "--music", "/music", "--data", "/data", "--out", "/out", "--workers", "2", "--interval-hours", "24", "--port", "8790"]
