import os
import json
import base64
import numpy as np
import sys
import io
import cv2
import pandas as pd
import gc
from typing import List
from tqdm import tqdm
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, FloatType, TimestampType
from PIL import Image
from ultralytics import YOLO
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from minio import Minio

# ==========================================
# PHẦN 1: CÁC HÀM HELPER
# ==========================================

def get_h_matrix():
    src_points = np.float32([[50, 270], [450, 270], [200, 200], [300, 200]])
    dst_points = np.float32([[-2.5, 5.0], [2.5, 5.0], [-1.0, 10.0], [1.0, 10.0]])
    H_matrix, _ = cv2.findHomography(src_points, dst_points)
    return H_matrix

def get_road_json_path(cam_id: str) -> str:
    try:
        if "_" not in cam_id: return None
        cam_num = int(cam_id.split('_')[1])
        if cam_num in [1, 2, 3, 11, 12, 13]: return "cam_01_02_03_11_12_13.json"
        elif cam_num in [4, 5, 14, 15]: return "cam_04_05_14_15.json"
        elif cam_num in [6, 7, 8, 16, 17, 18]: return "cam_06_07_08_16_17_18.json"
        elif cam_num in [9, 19]: return "cam_09_19.json"
        elif cam_num in [10, 20]: return "cam_10_20.json"
    except Exception:
        return None
    return None

def download_road_json_worker(minio_client, bucket_name, cam_id: str, local_path: str) -> bool:
    json_name = get_road_json_path(cam_id)
    if not json_name: return False
    object_name = f"road_geometry/{json_name}" 
    try:
        minio_client.fget_object(bucket_name, object_name, local_path)
        return True
    except Exception:
        return False

# ==========================================
# PHẦN 2: HÀM XỬ LÝ CHÍNH (Worker Side)
# ==========================================

def run_inference_worker(key, pdf: pd.DataFrame) -> pd.DataFrame:
    # 1. Config
    MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio-service:9000")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET_CONFIGS = "configs"
    MODEL_PATH = "/tmp/yolov8_traffic_best.pt" 
    
    # [TỐI ƯU] Lấy kích thước ảnh từ Env
    IMG_SIZE = int(os.environ.get("IMG_SIZE", 320))

    # 2. Init Resources
    try:
        minio_client = Minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=False)
        model = YOLO(MODEL_PATH)
        H_matrix = get_h_matrix()
    except Exception as e:
        print(f"Worker Init Error: {e}")
        return pd.DataFrame()

    # [QUAN TRỌNG] Khi group bởi 2 cột, key sẽ là một tuple (camera_id, batch_group)
    # Chúng ta chỉ cần lấy camera_id (phần tử đầu tiên)
    camera_id = key[0] 
    
    pdf = pdf.sort_values("timestamp")
    json_local = f"/tmp/road_{camera_id}.json"
    
    # 3. Download Road Geometry
    if not download_road_json_worker(minio_client, MINIO_BUCKET_CONFIGS, camera_id, json_local):
        road_multi_poly = Polygon() 
    else:
        try:
            with open(json_local, 'r') as f:
                data = json.load(f)
                road_polys = [s['points'] for s in data['shapes']]
            road_multi_poly = unary_union([Polygon(p) for p in road_polys if len(p) >= 3])
        except Exception:
            road_multi_poly = Polygon()
    
    track_positions = {}
    output_rows = []
    
    # Log progress bar cho worker này
    iterator = tqdm(pdf.iterrows(), total=len(pdf), desc=f"Cam {camera_id} Batch", file=sys.stdout, mininterval=10.0, ascii=True)

    # 4. Processing Loop
    for index, row in iterator:
        try:
            image_bytes = base64.b64decode(row["image"])
            with Image.open(io.BytesIO(image_bytes)) as img:
                image = img.convert('RGB')
                image_np = np.array(image)
            
            # Run Inference
            results = model.track(image_np, persist=True, verbose=False, imgsz=IMG_SIZE)
            
            boxes = []
            classes = []
            track_ids = []
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [-1] * len(boxes)
            elif "objects" in row and row["objects"] is not None:
                # Fallback nếu dùng metadata có sẵn (ít khi xảy ra với luồng này)
                for obj in row["objects"]:
                    xc, yc, wn, hn = obj["bbox"] 
                    h, w = image_np.shape[:2]
                    x1, y1 = (xc - wn/2) * w, (yc - hn/2) * h
                    x2, y2 = (xc + wn/2) * w, (yc + hn/2) * h
                    boxes.append([x1, y1, x2, y2])
                    classes.append(obj["class_id"])
                    track_ids.append(-1)

            for i, bbox in enumerate(boxes):
                x1, y1, x2, y2 = bbox
                
                # Tính overlap
                bbox_poly = box(x1, y1, x2, y2)
                overlap_area = 0.0
                if not road_multi_poly.is_empty:
                    overlap_area = bbox_poly.intersection(road_multi_poly).area
                
                # Perspective Transform
                u, v = (x1 + x2) / 2, y2
                point_vector = np.array([[[u, v]]], dtype=np.float32)
                world = cv2.perspectiveTransform(point_vector, H_matrix)[0][0]
                world_x, world_y = float(world[0]), float(world[1])

                track_id = int(track_ids[i]) if i < len(track_ids) else -1

                output_rows.append({
                    "camera_id": camera_id, "timestamp": row["timestamp"],
                    "class_id": int(classes[i]), "track_id": track_id,
                    "world_x": world_x, "world_y": world_y,
                    "overlap_area": overlap_area if overlap_area > 0 else 0.0,
                    "bbox_area": (x2 - x1) * (y2 - y1), "avg_speed": 0.0
                })
                
                if track_id != -1:
                    if track_id not in track_positions: track_positions[track_id] = []
                    track_positions[track_id].append((row["timestamp"], world_x, world_y))
            
            # [QUAN TRỌNG] Giải phóng bộ nhớ ngay sau mỗi frame
            del image_np
            del results
            
        except Exception as e:
            if index == 0: print(f"DEBUG ERROR: {e}")
            continue
    
    # Dọn dẹp bộ nhớ mạnh tay sau khi xong batch này
    gc.collect()

    # 5. Calculate Speed
    for track_id, pos_list in track_positions.items():
        if len(pos_list) < 2: continue
        pos_list.sort(key=lambda p: p[0])
        speeds = []
        for j in range(1, len(pos_list)):
            ts1, x1, y1 = pos_list[j-1]
            ts2, x2, y2 = pos_list[j]
            dt = (ts2 - ts1).total_seconds()
            if dt > 0:
                dist_km = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 1000.0
                speed = dist_km / (dt / 3600.0)
                if 0 < speed < 150: speeds.append(speed)
        
        avg_s = np.mean(speeds) if speeds else 0.0
        for row in output_rows:
            if row["track_id"] == track_id: row["avg_speed"] = float(avg_s)
    
    return pd.DataFrame(output_rows)

# ==========================================
# PHẦN 3: CLASS CONFIG (Driver Side)
# ==========================================
class InferenceProcessor:
    def __init__(self, spark: SparkSession, minio_endpoint: str, minio_access_key: str, minio_secret_key: str, minio_bucket: str, model_path: str):
        self.spark = spark
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.minio_bucket = minio_bucket

    def get_kafka_schema(self) -> StructType:
        object_schema = StructType([
            StructField("class_id", IntegerType()), 
            StructField("bbox", ArrayType(FloatType())) 
        ])
        return StructType([
            StructField("camera_id", StringType()), 
            StructField("timestamp", StringType()),  
            StructField("image_id", StringType()), 
            StructField("image", StringType()),  
            StructField("objects", ArrayType(object_schema))
        ])
    
    def read_from_kafka(self, kafka_broker: str, topic: str) -> DataFrame:
        raw_df = self.spark.read.format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .option("endingOffsets", "latest") \
            .load()
        
        parsed_df = raw_df.select(from_json(col("value").cast("string"), self.get_kafka_schema()).alias("data")).select("data.*")
        return parsed_df.withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))

    def download_road_json(self, cam_id: str, local_path: str) -> bool:
        client = Minio(self.minio_endpoint, self.minio_access_key, self.minio_secret_key, secure=False)
        return download_road_json_worker(client, self.minio_bucket, cam_id, local_path)

    def compute_road_area(self, road_polys: List[List]) -> float:
        polygons = [Polygon(poly) for poly in road_polys if len(poly) >= 3]
        if not polygons: return 0.0
        return unary_union(polygons).area

    def process_dataframe(self, df: DataFrame) -> DataFrame:
        output_schema = StructType([
            StructField("camera_id", StringType()), StructField("timestamp", TimestampType()),
            StructField("class_id", IntegerType()), StructField("track_id", IntegerType()),
            StructField("world_x", FloatType()), StructField("world_y", FloatType()),
            StructField("overlap_area", FloatType()), StructField("bbox_area", FloatType()),
            StructField("avg_speed", FloatType())
        ])
        
        # --- THAY ĐỔI ĐỂ TRÁNH OOM: CHIA DATA THEO 1 PHÚT ---
        # 1. Chuyển timestamp thành số giây (Long Type)
        # 2. Chia cho 60 (để nhóm thành từng phút) -> batch_group
        # Lưu ý: Nếu muốn nhóm 2 phút thì chia 120, 5 phút thì chia 300
        df_bucketed = df.withColumn("batch_group", (col("timestamp").cast("long") / 60).cast("int"))
        
        # GroupBy theo cả camera_id VÀ batch_group
        return df_bucketed.groupBy("camera_id", "batch_group").applyInPandas(run_inference_worker, schema=output_schema)