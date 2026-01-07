"""
YOLO model training module using Ultralytics
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO
import yaml


class YOLOModelTrainer:
    """Train YOLO model using Ultralytics"""
    
    def __init__(self, dataset_yaml: str, output_dir: str, model_size: str = "yolov8n.pt", resume_from: Optional[str] = None):
        """
        Initialize model trainer
        
        Args:
            dataset_yaml: Path to dataset YAML file
            output_dir: Directory to save training outputs
            model_size: YOLO model size (yolov8n.pt, yolov8s.pt, etc.)
            resume_from: Path to checkpoint file to resume training from (optional)
        """
        self.dataset_yaml = dataset_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        self.resume_from = resume_from
        
        # Use CPU only for training
        self.device = "cpu"
        print(f"Using device: {self.device} (CPU-only mode)")
        if self.resume_from:
            print(f"Will resume training from: {self.resume_from}")
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size for training
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results and metadata
        """
        # Determine model to load: resume from checkpoint if available, otherwise use base model
        model_path = self.resume_from if self.resume_from and Path(self.resume_from).exists() else self.model_size
        
        if self.resume_from and Path(self.resume_from).exists():
            print(f"Resuming training from checkpoint: {self.resume_from}")
        else:
            print(f"Starting fresh training with model: {self.model_size}")
        
        print(f"Dataset: {self.dataset_yaml}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
        
        # Load model (either from checkpoint or base model)
        model = YOLO(model_path)
        
        # Training arguments
        train_args = {
            'data': self.dataset_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            **kwargs
        }
        
        # Note: When loading model from checkpoint path, Ultralytics automatically resumes training
        # The 'resume' flag is only needed when resuming from the same output directory structure
        
        # Train model
        results = model.train(**train_args)
        
        # Get training results
        train_results_dir = self.output_dir / "train"
        weights_dir = train_results_dir / "weights"
        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"
        results_csv = train_results_dir / "results.csv"
        args_yaml = train_results_dir / "args.yaml"
        
        # Load metrics if available
        metrics = {}
        if results_csv.exists():
            try:
                import csv
                with open(results_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Get last row (final metrics)
                        final_metrics = rows[-1]
                        metrics = {
                            'final_epoch': int(final_metrics.get('epoch', epochs)),
                            'train_loss_box': float(final_metrics.get('train/box_loss', 0) or 0),
                            'train_loss_cls': float(final_metrics.get('train/cls_loss', 0) or 0),
                            'train_loss_dfl': float(final_metrics.get('train/dfl_loss', 0) or 0),
                            'val_loss_box': float(final_metrics.get('metrics/precision(B)', 0) or 0),
                            'precision': float(final_metrics.get('metrics/precision(B)', 0) or 0),
                            'recall': float(final_metrics.get('metrics/recall(B)', 0) or 0),
                            'mAP50': float(final_metrics.get('metrics/mAP50(B)', 0) or 0),
                            'mAP50-95': float(final_metrics.get('metrics/mAP50-95(B)', 0) or 0)
                        }
            except Exception as e:
                print(f"Error reading metrics: {e}")
        
        # Create metadata
        metadata = {
            'model_size': self.model_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'img_size': img_size,
            'device': self.device,
            'dataset_yaml': self.dataset_yaml,
            'best_weights': str(best_weights) if best_weights.exists() else None,
            'last_weights': str(last_weights) if last_weights.exists() else None,
            'args_yaml': str(args_yaml) if args_yaml.exists() else None,
            'metrics': metrics
        }
        
        return metadata
    
    def get_output_files(self) -> Dict[str, Path]:
        """
        Get paths to output files after training
        
        Returns:
            Dictionary with file type as key and Path as value
        """
        train_dir = self.output_dir / "train"
        weights_dir = train_dir / "weights"
        
        files = {
            'best_weights': weights_dir / "best.pt",
            'last_weights': weights_dir / "last.pt",
            'train_log': train_dir / "results.csv",  # Ultralytics saves as CSV, we'll rename to train.log
            'config_yaml': train_dir / "args.yaml",
            'metrics_json': train_dir / "metrics.json"
        }
        
        # Create metrics.json from results if it doesn't exist
        if not files['metrics_json'].exists() and files['train_log'].exists():
            self._create_metrics_json(files['train_log'], files['metrics_json'])
        
        return files
    
    def _create_metrics_json(self, results_csv: Path, metrics_json: Path):
        """Create metrics.json from results.csv"""
        try:
            import csv
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    final_metrics = rows[-1]
                    metrics = {
                        'training_completed': True,
                        'total_epochs': len(rows),
                        'final_metrics': {
                            k: float(v) if v and v.strip() else 0.0
                            for k, v in final_metrics.items()
                            if v and v.strip()
                        }
                    }
                    with open(metrics_json, 'w') as f:
                        json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error creating metrics.json: {e}")
            # Create empty metrics file
            with open(metrics_json, 'w') as f:
                json.dump({'training_completed': True, 'error': str(e)}, f, indent=2)

