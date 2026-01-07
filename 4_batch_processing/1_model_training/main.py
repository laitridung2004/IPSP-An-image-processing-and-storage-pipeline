"""
Main entry point for Spark batch processing YOLO training pipeline
"""
import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from config import Config
from data_processor import YOLODataProcessor
from model_trainer import YOLOModelTrainer
from minio_uploader import MinIOUploader


def create_spark_session() -> SparkSession:
    """Create and configure Spark session"""
    # Get available memory (default to 6Gi if not set, leaving some for system)
    import os
    available_memory = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    
    spark = SparkSession.builder \
        .appName(Config.SPARK_APP_NAME) \
        .master(Config.SPARK_MASTER) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", available_memory) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.executor.memory", available_memory) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "false") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("YOLO Model Training Pipeline - Spark Batch Processing")
    print("=" * 80)
    
    # Initialize directories
    Config.initialize_dirs()
    Config.DEVICE = Config.get_device()
    
    print(f"Configuration:")
    print(f"  Kafka Broker: {Config.KAFKA_BROKER}")
    print(f"  Topic: {Config.TOPIC_NAME}")
    print(f"  MinIO Bucket: {Config.MINIO_BUCKET}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Model Size: {Config.MODEL_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Image Size: {Config.IMG_SIZE}")
    print()
    
    # Create Spark session
    print("Creating Spark session...")
    spark = create_spark_session()
    
    try:
        # Step 1: Process data from Kafka
        print("\n[Step 1] Reading data from Kafka...")
        data_processor = YOLODataProcessor(spark, Config.DATA_DIR)
        kafka_df = data_processor.read_from_kafka(Config.KAFKA_BROKER, Config.TOPIC_NAME)
        
        print(f"Retrieved {kafka_df.count()} records from Kafka")
        
        if kafka_df.count() == 0:
            print("No data found in Kafka. Exiting.")
            return
        
        # Step 2: Convert to YOLO format
        print("\n[Step 2] Converting data to YOLO format...")
        processed_count = data_processor.process_dataframe(kafka_df)
        
        if processed_count == 0:
            print("No data was successfully processed. Exiting.")
            return
        
        # Step 3: Create dataset YAML
        print("\n[Step 3] Creating dataset configuration...")
        dataset_yaml = data_processor.create_dataset_yaml()
        print(f"Dataset YAML created: {dataset_yaml}")
        
        # Step 4: Check for existing checkpoint and download if available
        print("\n[Step 4] Checking for existing checkpoint...")
        checkpoint_dir = Path(Config.OUTPUT_DIR) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path_pt = checkpoint_dir / "last.pt"
        
        uploader = MinIOUploader(
            endpoint=Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            bucket=Config.MINIO_BUCKET,
            secure=Config.MINIO_SECURE
        )
        
        # Try to download last checkpoint from MinIO (check both .pt and .pth formats)
        # YOLO uses .pt format natively, so prefer .pt for resume
        checkpoint_object_pt = "yolo_training/last.pt"
        checkpoint_object_pth = "yolo_training/last.pth"
        resume_from = None
        
        if uploader.checkpoint_exists(checkpoint_object_pt):
            print(f"Found existing checkpoint in MinIO: {checkpoint_object_pt}")
            if uploader.download_checkpoint(str(checkpoint_path_pt), checkpoint_object_pt):
                resume_from = str(checkpoint_path_pt)
                print(f"Resuming training from checkpoint: {resume_from}")
            else:
                print("Failed to download checkpoint, starting fresh training")
        elif uploader.checkpoint_exists(checkpoint_object_pth):
            # If only .pth exists, download and rename to .pt for YOLO
            print(f"Found checkpoint in .pth format: {checkpoint_object_pth}")
            checkpoint_path_pth = checkpoint_dir / "last.pth"
            if uploader.download_checkpoint(str(checkpoint_path_pth), checkpoint_object_pth):
                import shutil
                shutil.copy2(checkpoint_path_pth, checkpoint_path_pt)
                resume_from = str(checkpoint_path_pt)
                print(f"Resuming training from checkpoint: {resume_from}")
            else:
                print("Failed to download checkpoint, starting fresh training")
        else:
            print("No existing checkpoint found, starting fresh training")
        
        # Step 5: Train YOLO model
        print("\n[Step 5] Training YOLO model...")
        trainer = YOLOModelTrainer(
            dataset_yaml=dataset_yaml,
            output_dir=Config.OUTPUT_DIR,
            model_size=Config.MODEL_SIZE,
            resume_from=resume_from
        )
        
        try:
            training_metadata = trainer.train(
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                img_size=Config.IMG_SIZE
            )
            
            print("\nTraining completed!")
            print(f"Best weights: {training_metadata.get('best_weights')}")
            print(f"Last weights: {training_metadata.get('last_weights')}")
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Step 6: Get output files
        print("\n[Step 6] Collecting output files...")
        output_files = trainer.get_output_files()
        
        # Handle Ultralytics output structure
        # Ultralytics saves best.pt and last.pt as .pt files, not .pth
        # We'll keep them as .pt or rename if needed
        final_files = {}
        for key, path in output_files.items():
            if path and path.exists():
                final_files[key] = path
                print(f"  Found: {key} -> {path}")
            else:
                print(f"  Missing: {key} -> {path}")
        
        # YOLO uses .pt format, but user requested .pth
        # Create .pth copies for compatibility
        import shutil
        train_dir = Path(Config.OUTPUT_DIR) / "train" / "weights"
        
        upload_files = {}
        
        # Handle best weights - create .pth version
        best_pt = train_dir / "best.pt"
        if best_pt.exists():
            best_pth = train_dir / "best.pth"
            shutil.copy2(best_pt, best_pth)
            upload_files['best_weights'] = best_pth
            print(f"  Created best.pth from best.pt")
        elif 'best_weights' in final_files and final_files['best_weights'].exists():
            upload_files['best_weights'] = final_files['best_weights']
        
        # Handle last weights - use .pt format for resume, uploader will also create .pth
        last_pt = train_dir / "last.pt"
        if last_pt.exists():
            upload_files['last_weights'] = last_pt  # Upload .pt (native format for resume)
            print(f"  Found last.pt for upload (will also create .pth copy)")
        elif 'last_weights' in final_files and final_files['last_weights'].exists():
            upload_files['last_weights'] = final_files['last_weights']
        
        # For train.log, copy results.csv if it exists
        if 'train_log' in final_files:
            upload_files['train_log'] = final_files['train_log']
        elif (Path(Config.OUTPUT_DIR) / "train" / "results.csv").exists():
            train_log_path = Path(Config.OUTPUT_DIR) / "train" / "train.log"
            shutil.copy2(Path(Config.OUTPUT_DIR) / "train" / "results.csv", train_log_path)
            upload_files['train_log'] = train_log_path
        
        # For config.yaml (args.yaml from Ultralytics)
        if 'config_yaml' in final_files and final_files['config_yaml'].exists():
            # Rename args.yaml to config.yaml
            config_yaml_path = train_dir.parent / "config.yaml"
            shutil.copy2(final_files['config_yaml'], config_yaml_path)
            upload_files['config_yaml'] = config_yaml_path
        elif (train_dir.parent / "args.yaml").exists():
            config_yaml_path = train_dir.parent / "config.yaml"
            shutil.copy2(train_dir.parent / "args.yaml", config_yaml_path)
            upload_files['config_yaml'] = config_yaml_path
        
        # For metrics.json
        if 'metrics_json' in final_files:
            upload_files['metrics_json'] = final_files['metrics_json']
        
        # Step 7: Upload to MinIO
        print("\n[Step 7] Uploading outputs to MinIO...")
        
        upload_results = uploader.upload_training_outputs(upload_files)
        
        # Upload dataset YAML as well
        uploader.upload_dataset_yaml(dataset_yaml)
        
        print("\nUpload results:")
        for file_type, success in upload_results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {file_type}")
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

