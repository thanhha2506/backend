# --- Stage 1: Build stage (cài dependencies)
FROM python:3.11-slim AS builder

# Cài các package cơ bản
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy requirements trước để tận dụng cache
COPY requirements.txt .

# Cài dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final image (nhẹ hơn)
FROM python:3.11-slim

# Tạo thư mục app
WORKDIR /app

# Copy từ builder: chỉ copy site-packages và pip
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy code app
COPY app/ ./app/

# Mở port (Railway sẽ tự bind $PORT)
ENV PORT=8000

# Start FastAPI với uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
