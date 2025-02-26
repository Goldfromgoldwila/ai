# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for torch, bitsandbytes, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port (Render assigns $PORT dynamically)
EXPOSE 8080

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT", "--workers", "1"]