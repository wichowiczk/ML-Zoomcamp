FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY predict.py .
COPY model_C=1.0.bin .

EXPOSE 5000

CMD ["waitress-serve", "--listen=0.0.0.0:5000", "predict:app"]