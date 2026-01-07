"""
Utility functions for YOLO training pipeline
"""
import json
from pathlib import Path
from typing import Dict, Any


def save_metadata(metadata: Dict[str, Any], output_path: Path):
    """
    Save metadata to JSON file
    
    Args:
        metadata: Dictionary with metadata
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(input_path: Path) -> Dict[str, Any]:
    """
    Load metadata from JSON file
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary with metadata
    """
    with open(input_path, 'r') as f:
        return json.load(f)








