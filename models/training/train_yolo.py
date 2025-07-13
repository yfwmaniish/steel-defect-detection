"""
Training Script for Steel Defect Detection System using YOLOv8
"""

import os
import logging
from typing import Dict, Optional
from pathlib import Path
import argparse
import yaml

from ultralytics import YOLO


logger = logging.getLogger(__name__)


def train_yolov8(config_path: str = "config/config.yaml", 
                 dataset_path: str = "data/labeled/dataset.yaml",
                 model_version: str = "yolov8n",
                 epochs: int = 100,
                 img_size: int = 720,
                 batch_size: int = 16,
                 device: str = "auto"):
    """
    Train YOLOv8 model on steel defect dataset.
    
    Args:
        config_path: Path to configuration YAML file
        dataset_path: Path to dataset YAML file
        model_version: YOLOv8 model version (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Training batch size
        device: Training device (auto, cpu, cuda)
    """
    
    # Check if dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset configuration not found: {dataset_path}")
        print(f"❌ Dataset configuration not found: {dataset_path}")
        print("Please run the dataset conversion script first:")
        print("python scripts/convert_neu_dataset.py")
        return
    
    # Load dataset configuration
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Check if training images exist
    train_images_path = Path(dataset_config['path']) / dataset_config['train']
    if not train_images_path.exists():
        logger.error(f"Training images not found: {train_images_path}")
        print(f"❌ Training images not found: {train_images_path}")
        return
    
    # Count training images
    train_images = list(train_images_path.glob('*.jpg')) + list(train_images_path.glob('*.png'))
    if not train_images:
        logger.error("No training images found")
        print("❌ No training images found")
        return
    
    print(f"🚀 Starting YOLOv8 training...")
    print(f"📊 Training images: {len(train_images)}")
    print(f"🤖 Model: {model_version}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📐 Image size: {img_size}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💻 Device: {device}")
    
    # Initialize YOLOv8 model
    model = YOLO(f"{model_version}.pt")
    
    # Training arguments
    train_args = {
        'data': str(dataset_path),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': 'models/training',
        'name': 'steel_defect_yolov8',
        'save': True,
        'save_period': 10,
        'cache': True,
        'workers': 4,
        'patience': 10,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'auto_augment': 'randaugment',
        'val': True,
        'plots': True,
        'exist_ok': True,
        'verbose': True
    }
    
    try:
        # Start training
        results = model.train(**train_args)
        
        # Save best model to our weights directory
        weights_dir = Path('models/weights')
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the best model
        best_model_path = Path(f'models/training/steel_defect_yolov8/weights/best.pt')
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, weights_dir / 'steel_defect_best.pt')
            print(f"✅ Best model saved to: {weights_dir / 'steel_defect_best.pt'}")
        
        # Print training results
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Training results:")
        print(f"   - Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   - Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   - Final Loss: {results.results_dict.get('train/box_loss', 'N/A')}")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train Steel Defect Detection Model using YOLOv8")
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='data/labeled/dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model version')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=720,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Training device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path('logs/train.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Start training
    train_yolov8(
        config_path=args.config,
        dataset_path=args.dataset,
        model_version=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

