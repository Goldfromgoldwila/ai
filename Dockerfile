FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \  # For torch
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install requirements
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY ./app.py app.py

# HF Spaces port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]