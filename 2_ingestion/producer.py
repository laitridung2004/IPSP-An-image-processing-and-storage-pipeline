import os
import time
import json
import lmdb
import pickle
import datetime
import threading
import base64
import logging
import numpy as np
from kafka import KafkaProducer
from minio import Minio

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configuration
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
TOPIC_NAME = "traffic_data"
NUM_PARTITIONS = 10 

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "bigdataproject") 
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "bigdataproject")
BUCKET_NAME = "traffic-data"
MINIO_PREFIX = "lambda_lmdb"

CAM_MODE = os.environ.get("CAM_MODE", "dev").lower()
FPS = 60

# Metrics
camera_metrics = {}
metrics_lock = threading.Lock()

def get_target_cameras():
    if CAM_MODE == "demo":
        return [f"cam_{i:02d}" for i in range(11, 21)]
    return [f"cam_{i:02d}" for i in range(1, 11)]

def stream_single_camera(cam_id, lmdb_path, producer):
    logging.info(f"[{cam_id}] Starting stream...")
    
    try:
        cam_num = int(cam_id.split('_')[1])
        target_partition = (cam_num - 1) % NUM_PARTITIONS
    except:
        target_partition = 0

    send_times = []
    total_blocked_time = 0
    frames_blocked = 0

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            count = 0
            for key, value in cursor:
                try:
                    frame_key = key.decode('utf-8')
                    
                    try:
                        record = pickle.loads(value)
                    except Exception as e:
                        logging.error(f"Pickle load error, trying msgpack: {e}")
                        import msgpack
                        record = msgpack.unpackb(value, raw=False)
                    
                    image_bytes = record.get('image')
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8') if image_bytes else None
                    
                    boxes = record.get('boxes')
                    if boxes is not None:
                        boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
                    else:
                        boxes_list = []
                    
                    objects = []
                    for box in boxes_list:
                        obj = {
                            'class_id': int(box[0]),
                            'bbox': [float(box[1]), float(box[2]), float(box[3]), float(box[4])]
                        }
                        objects.append(obj)
                    
                    current_timestamp = datetime.datetime.now().isoformat()

                    payload = {
                        'camera_id': cam_id,
                        'timestamp': current_timestamp,
                        'image_id': frame_key,
                        'image': image_base64,
                        'objects': objects
                    }
                    
                    send_start = time.time()
                    
                    producer.send(
                        TOPIC_NAME, 
                        key=cam_id, 
                        value=payload,
                        partition=target_partition
                    )
                    
                    send_duration = time.time() - send_start
                    send_times.append(send_duration)
                    
                    if send_duration > 0.05:
                        total_blocked_time += send_duration
                        frames_blocked += 1
                        
                        if send_duration > 0.2:
                            logging.warning(f"[{cam_id}] Frame {count}: blocked {send_duration*1000:.0f}ms")
                    
                    count += 1
                    if count % 50 == 0:
                        logging.info(f"[{cam_id}] -> Partition {target_partition}: sent {count} frames.")
                    
                    time.sleep(1.0 / FPS)

                except Exception as e:
                    logging.error(f"Error processing frame at {cam_id}: {e}")
                    continue
                    
        env.close()
        
        with metrics_lock:
            camera_metrics[cam_id] = {
                'total_frames': count,
                'send_times': send_times,
                'total_blocked_time': total_blocked_time,
                'frames_blocked': frames_blocked,
                'avg_send_time': sum(send_times) / len(send_times) if send_times else 0,
                'max_send_time': max(send_times) if send_times else 0,
                'min_send_time': min(send_times) if send_times else 0
            }
        
        logging.info(f"[{cam_id}] FINISHED - Sent {count} frames.")
        
    except Exception as e:
        logging.error(f"Thread error {cam_id}: {e}")

def run_producer():
    logging.info("Producer initializing (Multi-threaded / Fixed Partitioning)...")
    logging.info(f"MINIO_PREFIX: {MINIO_PREFIX}, CAM_MODE: {CAM_MODE}")
    
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        key_serializer=lambda k: k.encode('utf-8'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,
        batch_size=8192,
        linger_ms=10,
    )

    minio_client = Minio(
        MINIO_ENDPOINT, 
        access_key=MINIO_ACCESS_KEY, 
        secret_key=MINIO_SECRET_KEY, 
        secure=False
    )
    
    local_paths = {}
    target_cams = get_target_cameras()
    
    logging.info(f"Downloading data from MinIO bucket '{BUCKET_NAME}'...")
    
    for cam_id in target_cams:
        local_path = f"/tmp/{cam_id}.lmdb"
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        
        found = False
        for f in ["data.mdb", "lock.mdb"]:
            try:
                object_path = f"{MINIO_PREFIX}/{cam_id}.lmdb/{f}"
                local_file = f"{local_path}/{f}"
                minio_client.fget_object(BUCKET_NAME, object_path, local_file)
                found = True
            except Exception as e:
                logging.warning(f"Could not download {object_path}: {e}")
                continue
                
        if found:
            local_paths[cam_id] = local_path
            logging.info(f"Downloaded: {cam_id}")
        else:
            logging.warning(f"Data not found for {cam_id}")

    if not local_paths:
        logging.error("No cameras downloaded. Stopping.")
        return

    logging.info(f"Starting stream for {len(local_paths)} cameras...")
    
    overall_start = time.time()
    
    threads = []
    for cam_id, path in local_paths.items():
        t = threading.Thread(target=stream_single_camera, args=(cam_id, path, producer))
        t.start()
        threads.append(t)

    logging.info("Joining threads...")
    join_start = time.time()
    
    for t in threads:
        t.join()
    
    join_time = time.time() - join_start
    logging.info(f"Threads joined in {join_time:.2f}s")

    logging.info("Flushing producer...")
    flush_start = time.time()
    
    producer.flush()
    
    flush_time = time.time() - flush_start
    logging.info(f"Flush completed in {flush_time:.2f}s")
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print(" METRICS SUMMARY")
    print("="*80)
    
    if camera_metrics:
        total_frames = sum(m['total_frames'] for m in camera_metrics.values())
        total_blocked_time_all = sum(m['total_blocked_time'] for m in camera_metrics.values())
        total_frames_blocked = sum(m['frames_blocked'] for m in camera_metrics.values())
        
        all_send_times = []
        for m in camera_metrics.values():
            all_send_times.extend(m['send_times'])
        
        avg_send_time_all = sum(all_send_times) / len(all_send_times) if all_send_times else 0
        max_send_time_all = max(all_send_times) if all_send_times else 0
        min_send_time_all = min(all_send_times) if all_send_times else 0
        
        all_send_times_sorted = sorted(all_send_times)
        p50 = all_send_times_sorted[len(all_send_times_sorted)//2] if all_send_times_sorted else 0
        p95 = all_send_times_sorted[int(len(all_send_times_sorted)*0.95)] if all_send_times_sorted else 0
        p99 = all_send_times_sorted[int(len(all_send_times_sorted)*0.99)] if all_send_times_sorted else 0
        
        print(f"\n Overall Statistics:")
        print(f"   Total cameras:           {len(camera_metrics)}")
        print(f"   Total frames sent:       {total_frames:,}")
        print(f"   Total time:              {overall_time:.2f}s")
        print(f"   Overall throughput:      {total_frames/overall_time:.1f} frames/sec")
        
        print(f"\n Send Time Statistics:")
        print(f"   Average send time:       {avg_send_time_all*1000:.2f}ms")
        print(f"   Median (P50):            {p50*1000:.2f}ms")
        print(f"   P95:                     {p95*1000:.2f}ms")
        print(f"   P99:                     {p99*1000:.2f}ms")
        print(f"   Min send time:           {min_send_time_all*1000:.2f}ms")
        print(f"   Max send time:           {max_send_time_all*1000:.0f}ms")
        
        print(f"\n Blocking Statistics:")
        print(f"   Total blocked time:      {total_blocked_time_all:.2f}s")
        print(f"   Total frames blocked:    {total_frames_blocked:,} ({total_frames_blocked/total_frames*100:.1f}%)")
        
        print(f"\n Per-Camera Breakdown:")
        print(f"{'Camera':<10} {'Frames':<8} {'Avg(ms)':<10} {'Max(ms)':<10} {'Blocked':<10} {'Block%':<8}")
        print("-" * 80)
        
        for cam_id in sorted(camera_metrics.keys()):
            m = camera_metrics[cam_id]
            block_pct = m['frames_blocked'] / m['total_frames'] * 100 if m['total_frames'] > 0 else 0
            print(f"{cam_id:<10} {m['total_frames']:<8} {m['avg_send_time']*1000:<10.2f} "
                  f"{m['max_send_time']*1000:<10.0f} {m['frames_blocked']:<10} {block_pct:<8.1f}%")
        
    print("="*80)
    print("\nALL CAMERAS FINISHED.")

if __name__ == "__main__":
    run_producer()