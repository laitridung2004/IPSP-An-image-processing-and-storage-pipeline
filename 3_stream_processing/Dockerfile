FROM apache/spark:3.4.1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    pymongo \
    minio \
    numpy \
    opencv-python-headless \
    redis

COPY . .

CMD ["/opt/spark/bin/spark-submit", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.mongodb.spark:mongo-spark-connector_2.12:10.2.1", \
    "--conf", "spark.sql.execution.pythonUDF.arrow.enabled=false", \
    "--master", "local[*]", \
    "--py-files", "/app/util.py", \
    "/app/main.py"]