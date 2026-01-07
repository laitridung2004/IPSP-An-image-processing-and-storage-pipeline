import os
import sys
import json
import pymongo
import datetime  # [MỚI] Thêm thư viện datetime để xử lý lỗi
import matplotlib.pyplot as plt
from io import BytesIO
from typing import List, Dict

# Import đầy đủ các hàm cần thiết của Spark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import window, sum, count, count_distinct, avg, first, col, to_date, max, when, collect_list, struct, udf
from pyspark.sql.types import TimestampType, FloatType
from pyspark.conf import SparkConf

from minio import Minio
from util import InferenceProcessor

class Config:
    KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
    TOPIC_NAME = os.environ.get("TOPIC_NAME", "traffic_data")
    
    MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "bigdataproject")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "bigdataproject")
    MINIO_BUCKET_CONFIGS = "configs" 
    MINIO_BUCKET_DATA = "traffic-data"
    
    MODEL_OBJECT_KEY = "models/yolov8_traffic_best.pt"
    LOCAL_MODEL_PATH = "/tmp/yolov8_traffic_best.pt"

    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.environ.get("DB_NAME", "traffic_db")
    SPARK_APP_NAME = os.environ.get("SPARK_APP_NAME", "YOLO_Inference_Batch")

def create_spark_session() -> SparkSession:
    # --- CẤU HÌNH TỐI ƯU CHO MÁY YẾU (DELL E5550) ---
    conf = SparkConf()
    conf.set("spark.sql.shuffle.partitions", "2") 
    conf.set("spark.default.parallelism", "2")
    conf.set("spark.driver.memory", "800m")      
    conf.set("spark.executor.memory", "1g")
    conf.set("spark.memory.fraction", "0.6") 
    conf.set("spark.network.timeout", "600s") 

    spark = SparkSession.builder \
        .appName(Config.SPARK_APP_NAME) \
        .master(os.environ.get("SPARK_MASTER", "local[2]")) \
        .config(conf=conf) \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN") 
    return spark

# --- HÀM SỬA LỖI DATETIME Ở ĐÂY ---
def write_to_mongo(df: DataFrame, collection_name: str):
    mongo_uri = Config.MONGO_URI
    db_name = Config.DB_NAME
    
    def write_partition(partition):
        try:
            client = pymongo.MongoClient(mongo_uri)
            db = client[db_name]
            collection = db[collection_name]
            
            rows_to_insert = []
            for row in partition:
                row_dict = row.asDict()
                
                # [QUAN TRỌNG] Chuyển đổi datetime.date thành datetime.datetime
                # để Mongo có thể hiểu được
                for k, v in row_dict.items():
                    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
                        row_dict[k] = datetime.datetime.combine(v, datetime.time.min)
                
                rows_to_insert.append(row_dict)
            
            if rows_to_insert:
                collection.insert_many(rows_to_insert)
            
            client.close()
        except Exception as e:
            print(f"Error writing to Mongo: {e}")
            
    df.foreachPartition(write_partition)

def generate_heatmap(positions: List[Dict], camera_id: str, date_str: str, minio_client, bucket) -> str:
    if not positions: return None
    xs = [p['world_x'] for p in positions]
    ys = [p['world_y'] for p in positions]
    if len(xs) > 3000: 
        xs, ys = xs[::2], ys[::2]
        
    fig, ax = plt.subplots()
    try:
        ax.hist2d(xs, ys, bins=50, cmap='hot')
        ax.set_title(f"Traffic Heatmap {camera_id} {date_str}")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        object_name = f"reports/heatmaps/{camera_id}_{date_str.replace('-', '')}.png"
        minio_client.put_object(bucket, object_name, buf, len(buf.getvalue()), content_type="image/png")
        return f"http://{Config.MINIO_ENDPOINT}/{bucket}/{object_name}"
    except Exception as e:
        print(f"Heatmap Error: {e}")
        return None
    finally:
        plt.close(fig)

def download_model_from_minio(minio_client):
    local_path = Config.LOCAL_MODEL_PATH
    if os.path.exists(local_path): return local_path
    try:
        print("Downloading model from MinIO...")
        minio_client.fget_object(Config.MINIO_BUCKET_DATA, Config.MODEL_OBJECT_KEY, local_path)
        return local_path
    except Exception as e:
        print(f"ERROR downloading model: {e}"); sys.exit(1)

def main():
    print("=" * 60)
    print("YOLO Inference Pipeline - Survival Mode v12")
    print("=" * 60)
    
    try:
        minio_client = Minio(Config.MINIO_ENDPOINT, Config.MINIO_ACCESS_KEY, Config.MINIO_SECRET_KEY, secure=False)
        model_path = download_model_from_minio(minio_client)
    except Exception as e:
        print(f"MinIO/Model setup failed: {e}"); sys.exit(1)

    spark = create_spark_session()
    
    try:
        processor = InferenceProcessor(
            spark, Config.MINIO_ENDPOINT, Config.MINIO_ACCESS_KEY, Config.MINIO_SECRET_KEY, 
            Config.MINIO_BUCKET_CONFIGS, model_path=model_path
        )

        kafka_df = processor.read_from_kafka(Config.KAFKA_BROKER, Config.TOPIC_NAME)

        if kafka_df.count() == 0:
            print("No data found in Kafka."); return
        
        print("Running YOLO Inference & Logic Processing...")
        detections_df = processor.process_dataframe(kafka_df)
        
        detections_df.cache()
        count_detect = detections_df.count() 
        print(f"Total detections processed: {count_detect}")

        if count_detect == 0:
            print("No detections made."); return

        # 5. Density
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
        
        @udf(returnType=FloatType())
        def get_road_area_udf(cam_id):
            return float(road_areas_bc.value.get(cam_id, 1.0))
        
        frame_density_df = detections_df.groupBy("camera_id", "timestamp").agg(
            sum("overlap_area").alias("total_overlap")
        ).withColumn("road_area", get_road_area_udf(col("camera_id"))).withColumn(
            "density_pct", (col("total_overlap") / col("road_area")) * 100
        )
        
        print("Aggregating 1-min statistics...")
        window_spec = window("timestamp", "1 minute", "30 seconds")
        
        track_windows_df = detections_df.groupBy(window_spec.alias("window"), "camera_id", "class_id", "track_id").agg(first("avg_speed").alias("track_speed"))
        
        stats_1min_df = track_windows_df.groupBy("window", "camera_id", "class_id").agg(count("*").alias("unique_count"), avg("track_speed").alias("avg_speed"))
        
        class_map = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
        for cid, cname in class_map.items():
            stats_1min_df = stats_1min_df.withColumn(f"unique_{cname}_count", when(col("class_id") == cid, col("unique_count")).otherwise(0)) \
                .withColumn(f"avg_speed_{cname}_kmh", when(col("class_id") == cid, col("avg_speed")).otherwise(0.0))
        
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
        
        density_window_df = frame_density_df.groupBy(window("timestamp", "1 minute", "30 seconds").alias("window"), "camera_id").agg(avg("density_pct").alias("avg_density_pct")).withColumn("window_start", col("window.start")).withColumn("window_end", col("window.end"))
        
        final_stats_df = stats_1min_df.join(density_window_df, ["window_start", "window_end", "camera_id"], "inner").drop("window")
        
        write_to_mongo(final_stats_df, "traffic_stats_1min")
        
        print("Aggregating Daily statistics...")
        daily_df = detections_df.withColumn("date", to_date("timestamp"))
        daily_summary_df = daily_df.groupBy("date", "camera_id").agg(count_distinct("track_id").alias("total_vehicles_all_types"), avg("avg_speed").alias("avg_speed_day"))
        
        high_density_hours = frame_density_df \
            .withColumn("date", to_date("timestamp")) \
            .withColumn("hour", col("timestamp").cast(TimestampType()).cast("int") / 3600 % 24) \
            .where(col("density_pct") > 50) \
            .groupBy("date", "camera_id", "hour") \
            .count()

        peak_hour_df = high_density_hours.groupBy("date", "camera_id").agg(max("hour").alias("peak_hour"))
        congestion_hours_df = high_density_hours.groupBy("date", "camera_id").agg((sum("count") / 3600.0).alias("congestion_hours"))
        
        daily_summary_df = daily_summary_df.join(peak_hour_df, ["date", "camera_id"], "left").join(congestion_hours_df, ["date", "camera_id"], "left")
        
        write_to_mongo(daily_summary_df, "traffic_daily_summary")
        
        # 8. Heatmaps
        print("Generating Heatmaps...")
        positions_df = detections_df.select("camera_id", to_date("timestamp").alias("date"), struct("world_x", "world_y").alias("position")).groupBy("date", "camera_id").agg(collect_list("position").alias("positions"))
        
        for row in positions_df.collect():
            url = generate_heatmap(row["positions"], row["camera_id"], row["date"].isoformat(), minio_client, Config.MINIO_BUCKET_CONFIGS)
            if url:
                try:
                    pymongo.MongoClient(Config.MONGO_URI)[Config.DB_NAME]["traffic_heatmaps"].insert_one({
                        "date": row["date"].isoformat(), "camera_id": row["camera_id"], "heatmap_url": url, "description": "Heatmap"
                    })
                except Exception as e:
                    print(f"Mongo Insert Heatmap Error: {e}")
        
        # [QUAN TRỌNG] Xóa cache
        detections_df.unpersist()
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Critical Error: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()