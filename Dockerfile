FROM python:3.12-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + committed fixtures.
COPY . .

# Deterministic, CPU-only. Train then validate on container start.
CMD ["sh", "-c", "python main.py && python validate.py"]
