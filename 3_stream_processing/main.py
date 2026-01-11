import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf, get_json_object
from pyspark.sql.types import *
from util import process_traffic_analysis
from streaming_listener import PerformanceListener

# Configs
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://admin:password@mongo-service:27017")
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
TOPIC_NAME = os.environ.get("TOPIC_NAME", "traffic_data")

print(f"Starting Spark App: Kafka={KAFKA_BROKER}, Topic={TOPIC_NAME}")

# Spark Session
spark = SparkSession.builder \
    .appName("spark_stream_v1") \
    .config("spark.mongodb.write.connection.uri", MONGO_URI) \
    .config("spark.mongodb.write.database", "traffic_db") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Schema Definitions
object_schema = StructType([
    StructField("class_id", IntegerType()),
    StructField("bbox", ArrayType(FloatType())) 
])

kafka_schema = StructType([
    StructField("camera_id", StringType()),
    StructField("timestamp", StringType()),
    StructField("image_id", StringType()),
    StructField("objects", ArrayType(object_schema))
])

# UDF
process_udf = udf(process_traffic_analysis, StringType())

# Kafka Source
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Transformations
parsed_df = raw_df.select(
    from_json(col("value").cast("string"), kafka_schema).alias("data")
).select("data.*")

enriched_df = parsed_df.withColumn(
    "insights_json",
    process_udf(col("camera_id"), col("timestamp"), col("objects"))
)

final_df = enriched_df.select(
    col("camera_id"),
    col("timestamp"),
    get_json_object(col("insights_json"), "$.traffic_status").alias("traffic_status"),
    get_json_object(col("insights_json"), "$.density").cast("float").alias("density"),
    get_json_object(col("insights_json"), "$.alert_message").alias("alert_message"),
    get_json_object(col("insights_json"), "$.counts.motorbike").cast("int").alias("count_motorbike"),
    get_json_object(col("insights_json"), "$.counts.car").cast("int").alias("count_car"),
    get_json_object(col("insights_json"), "$.counts.bus").cast("int").alias("count_bus"),
    get_json_object(col("insights_json"), "$.counts.container").cast("int").alias("count_container"),
    get_json_object(col("insights_json"), "$.speeds.motorbike").cast("float").alias("speed_motorbike"),
    get_json_object(col("insights_json"), "$.speeds.car").cast("float").alias("speed_car"),
    get_json_object(col("insights_json"), "$.speeds.bus").cast("float").alias("speed_bus"),
    get_json_object(col("insights_json"), "$.speeds.container").cast("float").alias("speed_container")
)

print("Starting Streaming Query...")

# Start Query
query_monitor = final_df \
    .writeStream \
    .format("mongodb") \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint_spark_stream_v1") \
    .option("spark.mongodb.write.collection", "realtime_monitor") \
    .trigger(processingTime='15 seconds') \
    .start()

# Register Listener
listener = PerformanceListener(query_monitor)
spark.streams.addListener(listener)
print("Performance listener registered.")

try:
    query_monitor.awaitTermination()
except KeyboardInterrupt:
    print("Stopping query...")
    query_monitor.stop()