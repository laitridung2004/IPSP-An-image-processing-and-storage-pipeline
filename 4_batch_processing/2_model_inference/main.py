import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, sum, count_distinct, avg, first, col, to_date, max, min, count, when
from pyspark.sql.types import TimestampType
from util import InferenceProcessor
from minio import Minio
import pymongo
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import json
from pyspark.sql.functions import collect_list, struct

class Config:
    KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
    TOPIC_NAME = os.environ.get("TOPIC_NAME", "traffic_data")
    MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "bigdataproject")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "bigdataproject")
    MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "configs")
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.environ.get("DB_NAME", "traffic_db")
    SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
    SPARK_APP_NAME = os.environ.get("SPARK_APP_NAME", "YOLO_Inference_Batch")
    MODEL_OBJECT = "traffic-models/yolov8.pt"  

def create_spark_session() -> SparkSession:
    spark = SparkSession.builder \
        .appName(Config.SPARK_APP_NAME) \
        .master(Config.SPARK_MASTER) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def write_to_mongo(df: DataFrame, collection_name: str, mode="append"):
    mongo_uri = Config.MONGO_URI
    db_name = Config.DB_NAME
    
    def write_partition(partition):
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        for row in partition:
            collection.insert_one(row.asDict())
        client.close()
    
    df.foreachPartition(write_partition)

def generate_heatmap(positions: List[Dict], camera_id: str, date_str: str, minio_client, bucket) -> str:
    if not positions:
        return None
    
    xs = [p['world_x'] for p in positions]
    ys = [p['world_y'] for p in positions]
    
    fig, ax = plt.subplots()
    hist = ax.hist2d(xs, ys, bins=50, cmap='hot')
    ax.set_title(f"Traffic Heatmap {camera_id} {date_str}")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    object_name = f"reports/heatmaps/{camera_id}_{date_str.replace('-', '')}.png"
    minio_client.put_object(bucket, object_name, buf, len(buf.getvalue()), content_type="image/png")
    plt.close(fig)
    
    return f"http://{Config.MINIO_ENDPOINT}/{bucket}/{object_name}"

def main():
    print("=" * 80)
    print("YOLO Inference Pipeline - Spark Batch Processing")
    print("=" * 80)
    
    spark = create_spark_session()
    
    try:
        minio_client = Minio(Config.MINIO_ENDPOINT, Config.MINIO_ACCESS_KEY, Config.MINIO_SECRET_KEY, secure=False)
        
        processor = InferenceProcessor(spark, Config.MINIO_ENDPOINT, Config.MINIO_ACCESS_KEY, Config.MINIO_SECRET_KEY, Config.MINIO_BUCKET)
        kafka_df = processor.read_from_kafka(Config.KAFKA_BROKER, Config.TOPIC_NAME)
        
        if kafka_df.count() == 0:
            print("No data found. Exiting.")
            return
        
        detections_df = processor.process_dataframe(kafka_df)
        
        cameras = detections_df.select("camera_id").distinct().collect()
        road_areas = {}
        for cam in cameras:
            cam_id = cam["camera_id"]
            json_local = f"/tmp/road_{cam_id}.json"
            if processor.download_road_json(cam_id, json_local):
                with open(json_local, 'r') as f:
                    data = json.load(f)
                    road_polys = [s['points'] for s in data['shapes']]
                road_areas[cam_id] = processor.compute_road_area(road_polys)
            else:
                road_areas[cam_id] = 1.0
        
        road_areas_bc = spark.sparkContext.broadcast(road_areas)
        
        def get_road_area_udf(cam_id):
            return road_areas_bc.value.get(cam_id, 1.0)
        
        frame_density_df = detections_df.groupBy("camera_id", "timestamp").agg(
            sum("overlap_area").alias("total_overlap")
        ).withColumn("road_area", lit(get_road_area_udf(col("camera_id")))).withColumn(
            "density_pct", (col("total_overlap") / col("road_area")) * 100
        )
        
        window_spec = window("timestamp", "1 minute", "30 seconds")
        track_windows_df = detections_df.groupBy(
            window_spec.alias("window"),
            "camera_id",
            "class_id",
            "track_id"
        ).agg(
            first("avg_speed").alias("track_speed")
        )
        
        stats_1min_df = track_windows_df.groupBy(
            "window",
            "camera_id",
            "class_id"
        ).agg(
            count("*").alias("unique_count"),
            avg("track_speed").alias("avg_speed")
        )
        
        class_map = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
        for cid, cname in class_map.items():
            stats_1min_df = stats_1min_df.withColumn(
                f"unique_{cname}_count", when(col("class_id") == cid, col("unique_count")).otherwise(0)
            ).withColumn(
                f"avg_speed_{cname}_kmh", when(col("class_id") == cid, col("avg_speed")).otherwise(0.0)
            )
        
        stats_1min_df = stats_1min_df.groupBy("window", "camera_id").agg(
            sum("unique_motorcycle_count").alias("unique_motorcycle_count"),
            sum("unique_car_count").alias("unique_car_count"),
            sum("unique_bus_count").alias("unique_bus_count"),
            sum("unique_truck_count").alias("unique_truck_count"),
            avg("avg_speed_motorcycle_kmh").alias("avg_speed_motorcycle_kmh"),
            avg("avg_speed_car_kmh").alias("avg_speed_car_kmh"),
            avg("avg_speed_bus_kmh").alias("avg_speed_bus_kmh"),
            avg("avg_speed_truck_kmh").alias("avg_speed_truck_kmh")
        ).withColumn("window_start", col("window.start")).withColumn("window_end", col("window.end"))
        
        density_window_df = frame_density_df.groupBy(
            window("timestamp", "1 minute", "30 seconds").alias("window"),
            "camera_id"
        ).agg(
            avg("density_pct").alias("avg_density_pct")
        ).withColumn("window_start", col("window.start")).withColumn("window_end", col("window.end"))
        
        stats_1min_df = stats_1min_df.join(density_window_df, ["window_start", "window_end", "camera_id"], "inner")
        stats_1min_df = stats_1min_df.drop("window")
        
        write_to_mongo(stats_1min_df, "traffic_stats_1min")
        
        daily_df = detections_df.withColumn("date", to_date("timestamp"))
        daily_summary_df = daily_df.groupBy("date", "camera_id").agg(
            count_distinct("track_id").alias("total_vehicles_all_types"),
            avg("avg_speed").alias("avg_speed_day")
        )
        
        high_density_hours = frame_density_df.withColumn("hour", col("timestamp").cast(TimestampType()).cast("int") / 3600 % 24).where(col("density_pct") > 50).groupBy("date", "camera_id", "hour").count()
        peak_hour_df = high_density_hours.groupBy("date", "camera_id").agg(max("hour").alias("peak_hour"))
        congestion_hours_df = high_density_hours.groupBy("date", "camera_id").agg((count("*") / 3600.0).alias("congestion_hours"))
        
        daily_summary_df = daily_summary_df.join(peak_hour_df, ["date", "camera_id"], "left").join(congestion_hours_df, ["date", "camera_id"], "left")
        write_to_mongo(daily_summary_df, "traffic_daily_summary")
        
        positions_df = detections_df.select("camera_id", to_date("timestamp").alias("date"), struct("world_x", "world_y").alias("position")).groupBy("date", "camera_id").agg(collect_list("position").alias("positions"))
        positions_rows = positions_df.collect()
        heatmap_docs = []
        for row in positions_rows:
            url = generate_heatmap(row["positions"], row["camera_id"], row["date"].isoformat(), minio_client, Config.MINIO_BUCKET)
            if url:
                heatmap_docs.append({
                    "date": row["date"].isoformat(),
                    "camera_id": row["camera_id"],
                    "heatmap_url": url,
                    "description": f"Bản đồ nhiệt mật độ giao thông ngày {row['date'].isoformat()}"
                })
        
        if heatmap_docs:
            client = pymongo.MongoClient(Config.MONGO_URI)
            db = client[Config.DB_NAME]
            db["traffic_heatmaps"].insert_many(heatmap_docs)
            client.close()
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()