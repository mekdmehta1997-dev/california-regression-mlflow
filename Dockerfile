FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (Docker cache optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Default command: train + evaluate
CMD ["sh", "-c", "python src/train.py && python src/evaluate.py"]
