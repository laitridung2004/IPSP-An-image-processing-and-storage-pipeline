import cv2
import numpy as np
import json
import math
import os
import redis
import time
from datetime import datetime

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER", "bigdataproject")
MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD", "bigdataproject")
BUCKET_CONFIGS = "configs"

REDIS_HOST = os.environ.get("REDIS_HOST", "redis-service")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

IMG_W, IMG_H = 1280, 720
PIXELS_PER_METER = 20.0 
MAX_MATCH_DIST = 400.0 
MIN_MOVE_DIST_PX = 2.0

# Định nghĩa map class
CLASS_MAP = {
    0: "motorbike",
    1: "car",
    2: "bus",
    3: "container"
}

_road_masks_cache = {}
_road_area_cache = {} 

redis_client = None
minio_client = None

def get_redis():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_timeout=1)
        except Exception as e:
            print(f"Redis connection error: {e}")
            return None
    return redis_client

def get_minio_client():
    global minio_client
    if minio_client is None:
        from minio import Minio
        minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    return minio_client

def load_road_mask_data(camera_id, h=720, w=1280):
    """
    Trả về (mask, total_road_pixels)
    """
    if camera_id in _road_masks_cache:
        return _road_masks_cache[camera_id], _road_area_cache[camera_id]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        client = get_minio_client()
        mapping = {
            "cam_01_02_03_11_12_13.json": ["cam_01", "cam_02", "cam_03", "cam_11", "cam_12", "cam_13"],
            "cam_04_05_14_15.json": ["cam_04", "cam_05", "cam_14", "cam_15"],
            "cam_06_07_08_16_17_18.json": ["cam_06", "cam_07", "cam_08", "cam_16", "cam_17", "cam_18"],
            "cam_09_19.json": ["cam_09", "cam_19"],
            "cam_10_20.json": ["cam_10", "cam_20"]
        }
        target_file = next((f for f, cams in mapping.items() if camera_id in cams), None)
        if target_file:
            response = client.get_object(BUCKET_CONFIGS, f"road_geometry/{target_file}")
            config_data = json.loads(response.read().decode('utf-8'))
            for shape in config_data.get("shapes", []):
                if shape.get("label") in ["road", camera_id]:
                    pts = np.array(shape["points"], np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        
        road_pixels = np.count_nonzero(mask)
        if road_pixels == 0: 
            road_pixels = h * w
            
        _road_masks_cache[camera_id] = mask
        _road_area_cache[camera_id] = road_pixels
        
    except Exception as e:
        print(f"Error loading mask for {camera_id}: {e}")
        return np.ones((h, w), dtype=np.uint8), h*w

    return mask, _road_area_cache[camera_id]

def process_traffic_analysis(camera_id, timestamp_str, objects):
    try:
        try:
            ts_clean = timestamp_str.replace('Z', '')
            dt_obj = datetime.fromisoformat(ts_clean)
            t_curr = dt_obj.timestamp()
        except Exception as e:
            t_curr = time.time()

        _, road_area_px = load_road_mask_data(camera_id)

        counts = {k: 0 for k in CLASS_MAP.values()}
        speed_accumulators = {k: [] for k in CLASS_MAP.values()}
        total_vehicle_area_px = 0.0

        if not objects:
             empty_speed = {k: 0.0 for k in CLASS_MAP.values()}
             return json.dumps({
                "counts": counts,
                "speeds": empty_speed,
                "density": 0.0,
                "traffic_status": "NORMAL",
                "alert_message": None
            })

        curr_frame_objects_save = []
        
        for row_obj in objects:
            obj = row_obj.asDict() if hasattr(row_obj, "asDict") else row_obj
            bbox = obj.get('bbox')
            cls_id = obj.get('class_id')
            
            if bbox:
                nx, ny, nw, nh = bbox
                cx = int(nx * IMG_W)
                cy = int(ny * IMG_H)
                width_px = int(nw * IMG_W)
                height_px = int(nh * IMG_H)
                
                total_vehicle_area_px += (width_px * height_px)
                
                cls_name = CLASS_MAP.get(cls_id, "other")
                if cls_name in counts:
                    counts[cls_name] += 1
                
                curr_frame_objects_save.append({
                    "id": cls_id, 
                    "cx": cx, 
                    "cy": cy,
                    "cls_name": cls_name
                })

        density = 0.0
        if road_area_px > 0:
            density = total_vehicle_area_px / road_area_px

        r = get_redis()
        if r:
            redis_key = f"cam_speed:{camera_id}"
            prev_data_bytes = r.get(redis_key)
            
            new_state = {"timestamp": t_curr, "objects": curr_frame_objects_save}
            r.setex(redis_key, 30, json.dumps(new_state))

            if prev_data_bytes:
                prev_data = json.loads(prev_data_bytes)
                t_prev = prev_data.get('timestamp', 0)
                prev_objects = prev_data.get('objects', [])
                dt = t_curr - t_prev
                
                if 0.01 < dt < 20.0 and prev_objects:
                    for curr in curr_frame_objects_save:
                        min_dist = float('inf')
                        for prev in prev_objects:
                            if prev['id'] == curr['id']:
                                d = math.sqrt((curr['cx'] - prev['cx'])**2 + (curr['cy'] - prev['cy'])**2)
                                if d < min_dist:
                                    min_dist = d
                        
                        if min_dist < MAX_MATCH_DIST:
                            speed_kmh = 0.0
                            if min_dist >= MIN_MOVE_DIST_PX:
                                dist_meters = min_dist / PIXELS_PER_METER
                                speed_ms = dist_meters / dt
                                speed_kmh = speed_ms * 3.6
                            
                            if curr['cls_name'] in speed_accumulators:
                                speed_accumulators[curr['cls_name']].append(speed_kmh)

        final_speeds = {}
        for cls_name, speeds_list in speed_accumulators.items():
            if len(speeds_list) > 0:
                final_speeds[cls_name] = round(sum(speeds_list) / len(speeds_list), 2)
            else:
                final_speeds[cls_name] = 0.0

        status = "NORMAL"
        alert = None
        
        if density > 0.50:
            status = "JAMMED"
            alert = f"Jammed!"
        elif density > 0.25:
            status = "CROWDED"
            alert = f"Crowded!"

        result = {
            "counts": counts,
            "speeds": final_speeds,
            "density": round(density, 4),
            "traffic_status": status,
            "alert_message": alert
        }
        
        return json.dumps(result)

    except Exception as e:
        print(f"ERROR [{camera_id}]: {str(e)}")
        return json.dumps({
            "counts": {}, "speeds": {}, "density": 0.0, 
            "traffic_status": "ERROR", "alert_message": str(e)
        })