FROM python:3.11-slim

WORKDIR /app

COPY api/ /app/
COPY models/ /app/models

ENV MODEL_DIR=/app/models
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "api.py"]