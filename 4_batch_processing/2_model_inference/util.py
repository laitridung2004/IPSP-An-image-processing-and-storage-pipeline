"""
Inference processing module for YOLO tracking and traffic insights
"""
import os
import json
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, FloatType, TimestampType
from PIL import Image
import io
import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from minio import Minio
import pandas as pd
from datetime import timedelta

class InferenceProcessor:  
    def __init__(self, spark: SparkSession, minio_endpoint: str, minio_access_key: str, minio_secret_key: str, minio_bucket: str, model_path: str = "/app/models/yolov8.pt"):
        self.spark = spark
        self.minio_client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)
        self.minio_bucket = minio_bucket
        self.model = YOLO(model_path)
        src_points = np.float32([[50, 270], [450, 270], [200, 200], [300, 200]])
        dst_points = np.float32([[-2.5, 5.0], [2.5, 5.0], [-1.0, 10.0], [1.0, 10.0]])
        self.H_matrix, _ = cv2.findHomography(src_points, dst_points)
    
    def get_kafka_schema(self) -> StructType:
        """Updated schema matching provided: includes timestamp and objects"""
        object_schema = StructType([
            StructField("class_id", IntegerType()),
            StructField("bbox", ArrayType(FloatType()))
        ])
        
        kafka_schema = StructType([
            StructField("camera_id", StringType()),
            StructField("timestamp", StringType()),  
            StructField("image_id", StringType()),
            StructField("image", StringType()),  
            StructField("objects", ArrayType(object_schema))
        ])
        return kafka_schema
    
    def read_from_kafka(self, kafka_broker: str, topic: str) -> DataFrame:
        """Read and parse; timestamp directly parsed"""
        raw_df = self.spark.read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .option("endingOffsets", "latest") \
            .load()
        
        schema = self.get_kafka_schema()
        parsed_df = raw_df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        parsed_df = parsed_df.withColumn("timestamp", to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
        
        return parsed_df
    
    def get_road_json_path(self, cam_id: str) -> str:
        try:
            cam_num = int(cam_id.split('_')[1])
            if cam_num in [1, 2, 3, 11, 12, 13]:
                return "cam_01_02_03_11_12_13.json"
            elif cam_num in [4, 5, 14, 15]:
                return "cam_04_05_14_15.json"
            elif cam_num in [6, 7, 8, 16, 17, 18]:
                return "cam_06_07_08_16_17_18.json"
            elif cam_num in [9, 19]:
                return "cam_09_19.json"
            elif cam_num in [10, 20]:
                return "cam_10_20.json"
        except:
            return None
        return None
    
    def download_road_json(self, cam_id: str, local_path: str) -> bool:
        json_name = self.get_road_json_path(cam_id)
        if not json_name:
            return False
        object_name = f"road_segmentation/{json_name}"
        try:
            self.minio_client.fget_object(self.minio_bucket, object_name, local_path)
            return True
        except Exception as e:
            print(f"Error downloading road JSON for {cam_id}: {e}")
            return False
    
    def compute_road_area(self, road_polys: List[List]) -> float:
        polygons = [Polygon(poly) for poly in road_polys if len(poly) >= 3]
        if not polygons:
            return 0.0
        multi_poly = unary_union(polygons)
        return multi_poly.area
    
    def process_group(self, key, pdf: pd.DataFrame) -> pd.DataFrame:
        camera_id = key[0]
        pdf = pdf.sort_values("timestamp")
        
        json_local = f"/tmp/road_{camera_id}.json"
        if not self.download_road_json(camera_id, json_local):
            return pd.DataFrame()
        with open(json_local, 'r') as f:
            data = json.load(f)
            road_polys = [s['points'] for s in data['shapes']]
        road_multi_poly = unary_union([Polygon(p) for p in road_polys if len(p) >= 3])
        road_area = road_multi_poly.area
        
        track_positions = {}
        
        output_rows = []
        for _, row in pdf.iterrows():
            image_bytes = base64.b64decode(row["image"])
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            
            results = self.model.track(image_np, persist=True, verbose=False)
            if results[0].boxes is None:
                for obj in row["objects"]:
                    class_id = obj["class_id"]
                    xc, yc, wn, hn = obj["bbox"]
                    x1 = (xc - wn/2) * w
                    y1 = (yc - hn/2) * h
                    x2 = (xc + wn/2) * w
                    y2 = (yc + hn/2) * h
                    track_id = -1
                    bbox = [x1, y1, x2, y2]
                    
                    bbox_poly = box(x1, y1, x2, y2)
                    intersection = bbox_poly.intersection(road_multi_poly)
                    overlap_area = intersection.area
                    
                    u_center = (x1 + x2) / 2
                    v_bottom = y2
                    world = cv2.perspectiveTransform(np.float32([[u_center, v_bottom]]), self.H_matrix)[0][0]
                    world_x, world_y = float(world[0]), float(world[1])
                    
                    output_rows.append({
                        "camera_id": camera_id,
                        "timestamp": row["timestamp"],
                        "class_id": class_id,
                        "track_id": track_id,
                        "world_x": world_x,
                        "world_y": world_y,
                        "overlap_area": overlap_area if overlap_area > 0 else 0.0,
                        "bbox_area": (x2 - x1) * (y2 - y1)
                    })
                    
                    if track_id not in track_positions:
                        track_positions[track_id] = []
                    track_positions[track_id].append((row["timestamp"], world_x, world_y, class_id))
                continue
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [-1] * len(boxes)
            
            for i, bbox in enumerate(boxes):
                class_id = classes[i]
                track_id = track_ids[i]
                x1, y1, x2, y2 = bbox
                
                bbox_poly = box(x1, y1, x2, y2)
                intersection = bbox_poly.intersection(road_multi_poly)
                overlap_area = intersection.area
                
                u_center = (x1 + x2) / 2
                v_bottom