FROM apache/spark:3.4.1
USER root

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch torchvision

RUN pip3 install --no-cache-dir \
    ultralytics \
    minio \
    Pillow \
    PyYAML \
    opencv-python-headless \
    shapely \
    pymongo \
    matplotlib

RUN pip3 install --no-cache-dir pyspark==3.4.1

RUN mkdir -p /app/models
COPY models/yolov8.pt /app/models/yolov8.pt

COPY . .

CMD ["/opt/spark/bin/spark-submit", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1", \
    "--conf", "spark.sql.execution.pythonUDF.arrow.enabled=false", \
    "--driver-memory", "2g", \
    "--executor-memory", "2g", \
    "main.py"]