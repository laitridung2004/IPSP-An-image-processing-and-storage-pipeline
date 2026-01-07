FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install kafka-python minio lmdb msgpack numpy

COPY producer.py .

CMD ["python", "producer.py"]