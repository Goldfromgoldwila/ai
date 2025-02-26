FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache this layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (changes more often, so last)
COPY app.py .

EXPOSE 8080

CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1