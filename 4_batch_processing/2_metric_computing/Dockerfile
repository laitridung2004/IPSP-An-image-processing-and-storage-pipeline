FROM apache/spark:3.4.1
USER root

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip

RUN pip3 install "typing-extensions<4.10.0" --no-cache-dir

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --no-cache-dir "numpy<2.0.0" 

RUN pip3 install --no-cache-dir \
    "pandas<2.1.0" \
    pyarrow \
    ultralytics \
    minio \
    Pillow \
    PyYAML \
    opencv-python-headless \
    shapely \
    pymongo \
    matplotlib \
    lap \
    tqdm

RUN pip3 install --no-cache-dir pyspark==3.4.1

COPY . .

CMD ["/opt/spark/bin/spark-submit", \
    "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1", \
    "--conf", "spark.sql.execution.pythonUDF.arrow.enabled=false", \
    "main.py"]