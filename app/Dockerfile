FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements from root
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only app folder
COPY app/ ./app/

# Copy only model_registry folder
COPY model_registry/ ./model_registry/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]