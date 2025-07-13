"""
Advanced Training Script for Steel Defect Detection with Fine-tuning
"""

import os
import logging
from typing import Dict, Optional
from pathlib import Path
import argparse
import yaml
import numpy as np
from collections import Counter

from ultralytics import YOLO


logger = logging.getLogger(__name__)


def analyze_dataset(dataset_path: str):
    """Analyze dataset class distribution for informed training."""
    print("📊 Analyzing dataset distribution...")
    
    # Count class occurrences in label files
    class_counts = Counter()
    labels_dir = Path(dataset_path).parent / "train" / "labels"
    
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(labels_dir, filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
    
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    print("Class distribution:")
    total_instances = sum(class_counts.values())
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_instances) * 100
        print(f"  Class {class_id} ({class_names[class_id]}): {count} instances ({percentage:.1f}%)")
    
    return class_counts


def get_optimized_hyperparameters(model_version: str = "yolov8n"):
    """Get optimized hyperparameters based on our dataset analysis."""
    
    # Base hyperparameters optimized for steel defect detection
    base_params = {
        # Data augmentation - more aggressive for better generalization
        'hsv_h': 0.02,      # Slight hue variation (steel has consistent color)
        'hsv_s': 0.5,       # Moderate saturation changes
        'hsv_v': 0.3,       # Brightness variations for different lighting
        'degrees': 5.0,     # Small rotations (steel pieces might be rotated)
        'translate': 0.2,   # Translation augmentation
        'scale': 0.8,       # Scale variations
        'shear': 2.0,       # Small shear transformations
        'perspective': 0.0001,  # Minimal perspective (steel surfaces are mostly flat)
        'flipud': 0.0,      # No vertical flips (gravity affects defects)
        'fliplr': 0.5,      # Horizontal flips OK
        'mosaic': 1.0,      # Mosaic augmentation
        'mixup': 0.1,       # Small amount of mixup
        'copy_paste': 0.2,  # Copy-paste augmentation for small defects
        'erasing': 0.3,     # Random erasing
        
        # Training parameters
        'lr0': 0.005,       # Lower initial learning rate for fine-tuning
        'lrf': 0.01,        # Final learning rate
        'momentum': 0.9,    # Slightly lower momentum
        'weight_decay': 0.001,  # Increased weight decay for regularization
        'warmup_epochs': 5,     # More warmup epochs
        'warmup_momentum': 0.5, # Lower warmup momentum
        'warmup_bias_lr': 0.05, # Lower warmup bias learning rate
        
        # Loss function weights - adjusted for steel defects
        'box': 7.5,         # Box regression loss weight
        'cls': 1.0,         # Classification loss weight (increased)
        'dfl': 1.5,         # Distribution focal loss weight
        
        # Optimization
        'optimizer': 'AdamW',
        'close_mosaic': 15,  # Close mosaic augmentation later
        'amp': False,       # Disable AMP for GTX 1650 stability
        
        # Validation
        'val': True,
        'plots': True,
        'save_period': 5,   # Save more frequently
    }
    
    return base_params


def train_yolov8_advanced(
    config_path: str = "config/config.yaml", 
    dataset_path: str = "data/labeled/dataset.yaml",
    model_version: str = "yolov8n",
    epochs: int = 150,
    img_size: int = 640,
    batch_size: int = 8,
    device: str = "0",
    resume_from: str = None,
    freeze_layers: int = 0
):
    """
    Advanced training with optimized hyperparameters and techniques.
    """
    
    # Check if dataset exists
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset configuration not found: {dataset_path}")
        print(f"❌ Dataset configuration not found: {dataset_path}")
        return
    
    # Analyze dataset
    class_counts = analyze_dataset(str(dataset_path))
    
    # Load dataset configuration
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Check training images
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
    
    print(f"🚀 Starting Advanced YOLOv8 Fine-tuning...")
    print(f"📊 Training images: {len(train_images)}")
    print(f"🤖 Model: {model_version}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📐 Image size: {img_size}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💻 Device: {device}")
    if resume_from:
        print(f"📋 Resuming from: {resume_from}")
    if freeze_layers > 0:
        print(f"❄️ Freezing first {freeze_layers} layers")
    
    # Initialize model
    if resume_from and os.path.exists(resume_from):
        print(f"📋 Loading existing model: {resume_from}")
        model = YOLO(resume_from)
    else:
        print(f"🆕 Starting from pretrained {model_version}")
        model = YOLO(f"{model_version}.pt")
    
    # Freeze layers if specified
    if freeze_layers > 0:
        for i, (name, param) in enumerate(model.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False
                print(f"❄️ Frozen layer: {name}")
    
    # Get optimized hyperparameters
    optimized_params = get_optimized_hyperparameters(model_version)
    
    # Training arguments
    train_args = {
        'data': str(dataset_path),
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': 'models/training',
        'name': 'steel_defect_yolov8_advanced',
        'save': True,
        'cache': 'disk',  # Use disk cache for consistency
        'workers': 4,
        'patience': 20,   # Increased patience for better convergence
        'exist_ok': True,
        'verbose': True,
        **optimized_params
    }
    
    try:
        print("\n🎯 Starting advanced training with optimized hyperparameters...")
        print("🔧 Key optimizations:")
        print(f"   - Enhanced data augmentation for steel defects")
        print(f"   - Optimized learning rate schedule")
        print(f"   - Balanced loss weights")
        print(f"   - Disk caching for consistency")
        print(f"   - Extended patience for convergence")
        
        # Start training
        results = model.train(**train_args)
        
        # Save best model to our weights directory
        weights_dir = Path('models/weights')
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the best model
        best_model_path = Path(f'models/training/steel_defect_yolov8_advanced/weights/best.pt')
        if best_model_path.exists():
            import shutil
            shutil.copy(best_model_path, weights_dir / 'steel_defect_advanced.pt')
            print(f"✅ Advanced model saved to: {weights_dir / 'steel_defect_advanced.pt'}")
        
        # Print training results
        print(f"\n🎉 Advanced training completed successfully!")
        print(f"📈 Training results:")
        if hasattr(results, 'results_dict'):
            print(f"   - Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   - Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # Run validation on the advanced model
        print(f"\n🔬 Running validation on advanced model...")
        val_results = model.val(data=str(dataset_path))
        
        logger.info("Advanced training completed successfully")
        
        return model, results
        
    except Exception as e:
        logger.error(f"Advanced training failed: {str(e)}")
        print(f"❌ Advanced training failed: {str(e)}")
        raise


def hyperparameter_search(dataset_path: str, trials: int = 3):
    """
    Perform hyperparameter search to find optimal settings.
    """
    print(f"🔍 Starting hyperparameter search with {trials} trials...")
    
    # Define hyperparameter search space
    search_space = {
        'lr0': [0.001, 0.005, 0.01],
        'weight_decay': [0.0005, 0.001, 0.002],
        'momentum': [0.9, 0.937, 0.95],
        'box': [7.5, 10.0, 12.5],
        'cls': [0.5, 1.0, 1.5],
    }
    
    best_map = 0
    best_params = {}
    
    for trial in range(trials):
        print(f"\n🧪 Trial {trial + 1}/{trials}")
        
        # Sample random hyperparameters
        params = {}
        for param, values in search_space.items():
            params[param] = np.random.choice(values)
        
        print(f"Testing parameters: {params}")
        
        try:
            # Train with these parameters
            model = YOLO("yolov8n.pt")
            
            train_args = {
                'data': dataset_path,
                'epochs': 30,  # Shorter epochs for search
                'imgsz': 512,
                'batch': 4,
                'device': '0',
                'project': 'models/hp_search',
                'name': f'trial_{trial}',
                'save': False,  # Don't save intermediate models
                'verbose': False,
                **params
            }
            
            results = model.train(**train_args)
            
            # Get validation mAP
            val_results = model.val(data=dataset_path, verbose=False)
            current_map = val_results.box.map50
            
            print(f"   mAP50: {current_map:.3f}")
            
            if current_map > best_map:
                best_map = current_map
                best_params = params.copy()
                print(f"   🏆 New best mAP50: {best_map:.3f}")
            
        except Exception as e:
            print(f"   ❌ Trial failed: {str(e)}")
            continue
    
    print(f"\n🎯 Hyperparameter search completed!")
    print(f"🏆 Best mAP50: {best_map:.3f}")
    print(f"🔧 Best parameters: {best_params}")
    
    return best_params


def main():
    parser = argparse.ArgumentParser(description="Advanced Steel Defect Detection Training")
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='data/labeled/dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model version')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Training device')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--freeze-layers', type=int, default=0,
                        help='Number of layers to freeze for fine-tuning')
    parser.add_argument('--hp-search', action='store_true',
                        help='Perform hyperparameter search')
    parser.add_argument('--hp-trials', type=int, default=3,
                        help='Number of hyperparameter search trials')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path('logs/train_advanced.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    if args.hp_search:
        # Perform hyperparameter search
        best_params = hyperparameter_search(args.dataset, args.hp_trials)
        print(f"\n💾 Save these parameters for your next training run:")
        print(f"   {best_params}")
    else:
        # Start advanced training
        model, results = train_yolov8_advanced(
            config_path=args.config,
            dataset_path=args.dataset,
            model_version=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device,
            resume_from=args.resume_from,
            freeze_layers=args.freeze_layers
        )


if __name__ == "__main__":
    main()
