"""
Configuration management for YOLO model training pipeline
"""
import os
from typing import Optional


class Config:
    """Configuration class for training pipeline"""
    
    # Kafka Configuration
    KAFKA_BROKER: str = os.environ.get("KAFKA_BROKER", "my-kafka:9092")
    TOPIC_NAME: str = os.environ.get("TOPIC_NAME", "traffic_data")
    
    # MinIO Configuration
    MINIO_ENDPOINT: str = os.environ.get("MINIO_ENDPOINT", "my-minio:9000")
    MINIO_ACCESS_KEY: str = os.environ.get("MINIO_ACCESS_KEY", "bigdataproject")
    MINIO_SECRET_KEY: str = os.environ.get("MINIO_SECRET_KEY", "bigdataproject")
    MINIO_BUCKET: str = os.environ.get("MINIO_BUCKET", "configs")
    MINIO_SECURE: bool = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    
    # Spark Configuration
    SPARK_MASTER: str = os.environ.get("SPARK_MASTER", "local[*]")
    SPARK_APP_NAME: str = os.environ.get("SPARK_APP_NAME", "YOLO_Training_Batch")
    
    # Training Configuration
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "16"))
    EPOCHS: int = int(os.environ.get("EPOCHS", "5"))
    IMG_SIZE: int = int(os.environ.get("IMG_SIZE", "640"))
    DATA_DIR: str = os.environ.get("DATA_DIR", "/tmp/yolo_training_data")
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "/tmp/yolo_training_output")
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "weights")
    
    # Model Configuration
    MODEL_SIZE: str = os.environ.get("MODEL_SIZE", "yolov8n.pt")  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    
    # GPU Configuration
    DEVICE: Optional[str] = None  # Will be auto-detected
    
    @classmethod
    def get_device(cls) -> str:
        """Return CPU device (CPU-only training)"""
        return "cpu"
    
    @classmethod
    def initialize_dirs(cls):
        """Initialize necessary directories"""
        import os
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)

