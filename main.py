"""
Main CLI for Steel Defect Detection System
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.detection.detection_pipeline import DetectionPipeline
from src.utils.config import Config


def main():
    """Main entry point for the steel defect detection system."""
    parser = argparse.ArgumentParser(
        description="Steel Defect Detection System - AI-powered defect detection for steel sheets"
    )
    
    parser.add_argument(
        'command',
        choices=['detect', 'batch', 'pair', 'train', 'test'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input image path or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--image-id',
        type=str,
        help='Image ID for camera pair processing'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save detection visualizations'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        help='Confidence threshold for detections'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        help='IoU threshold for NMS'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.command in ['detect', 'batch', 'pair'] and not args.input:
        print("Error: --input is required for detection commands")
        sys.exit(1)
    
    if args.command == 'pair' and not args.image_id:
        print("Error: --image-id is required for pair command")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = DetectionPipeline(args.config)
        
        # Update thresholds if specified
        if args.confidence or args.iou:
            pipeline.detector.update_thresholds(args.confidence, args.iou)
        
        # Execute command
        if args.command == 'detect':
            detect_single(pipeline, args)
        elif args.command == 'batch':
            detect_batch(pipeline, args)
        elif args.command == 'pair':
            detect_pair(pipeline, args)
        elif args.command == 'train':
            train_model(args)
        elif args.command == 'test':
            test_system(pipeline, args)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def detect_single(pipeline: DetectionPipeline, args):
    """Detect defects in a single image."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing single image: {input_path}")
    
    # Process image
    result = pipeline.process_single_image(input_path)
    
    if result['success']:
        print(f"✅ Processing completed successfully!")
        print(f"   - Processing time: {result['processing_time']:.2f}s")
        print(f"   - Defects found: {result['num_defects']}")
        
        # Print defect details
        if result['detections']:
            print("\nDefect Details:")
            for i, detection in enumerate(result['detections'], 1):
                print(f"   {i}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
                bbox = detection['bbox']
                print(f"      Location: ({bbox['x']:.0f}, {bbox['y']:.0f}) "
                      f"Size: {bbox['width']:.0f}x{bbox['height']:.0f}")
        
        # Save visualization if requested
        if args.visualize:
            output_dir = Path(args.output) if args.output else Path("results/visualizations")
            pipeline._save_visualization(input_path, result, output_dir)
            print(f"   - Visualization saved to: {output_dir}")
    
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")


def detect_batch(pipeline: DetectionPipeline, args):
    """Detect defects in a batch of images."""
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else Path("results")
    
    print(f"Processing batch of images from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process batch
    results = pipeline.process_batch(
        input_dir,
        output_dir,
        save_visualizations=args.visualize
    )
    
    if results:
        # Save results to CSV
        csv_path = pipeline.save_results_to_csv()
        
        # Get summary
        summary = pipeline.get_detection_summary()
        
        print(f"\n✅ Batch processing completed!")
        print(f"   - Images processed: {len(results)}")
        print(f"   - Total defects found: {summary['total_detections']}")
        print(f"   - Results saved to: {csv_path}")
        
        # Print defect type summary
        if summary['defect_types']:
            print("\nDefect Type Summary:")
            for defect_type, count in summary['defect_types'].items():
                print(f"   - {defect_type}: {count}")
        
        print(f"   - Average confidence: {summary.get('average_confidence', 0):.2f}")
    
    else:
        print("❌ No images processed")


def detect_pair(pipeline: DetectionPipeline, args):
    """Detect defects in a camera pair (top and bottom images)."""
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else Path("results")
    
    print(f"Processing camera pair for image ID: {args.image_id}")
    
    # Process camera pair
    result = pipeline.process_camera_pair(args.image_id, input_dir, output_dir)
    
    print(f"\n✅ Camera pair processing completed!")
    print(f"   - Image ID: {result['image_id']}")
    print(f"   - Total defects: {result['total_defects']}")
    
    # Print results for each camera
    for camera in ['top', 'bottom']:
        camera_result = result[f'{camera}_result']
        if camera_result:
            if camera_result['success']:
                print(f"   - {camera.title()} camera: {camera_result['num_defects']} defects")
            else:
                print(f"   - {camera.title()} camera: Processing failed")
        else:
            print(f"   - {camera.title()} camera: Image not found")


def train_model(args):
    """Train the steel defect detection model."""
    print("🚀 Starting model training...")
    
    config = Config(args.config)
    
    # Check if labeled data exists
    labeled_data_path = Path(config.get('data.labeled_data_path', 'data/labeled'))
    if not labeled_data_path.exists() or not any(labeled_data_path.iterdir()):
        print(f"❌ No labeled data found in: {labeled_data_path}")
        print("Please prepare labeled dataset before training.")
        print("\n📋 To prepare dataset:")
        print("1. Place images in data/labeled/images/")
        print("2. Place YOLO format labels in data/labeled/labels/")
        print("3. Create dataset.yaml configuration file")
        sys.exit(1)
    
    # Import and run training
    try:
        from models.training.train_yolo import train_yolov8
        train_yolov8(args.config)
        print("✅ Training completed successfully!")
    except ImportError as e:
        print(f"❌ Training module not found: {e}")
        print("Training functionality requires labeled dataset.")


def test_system(pipeline: DetectionPipeline, args):
    """Test the system with sample data."""
    print("🧪 Testing Steel Defect Detection System...")
    
    # Test configuration
    config_info = pipeline.detector.get_model_info()
    print(f"✅ Configuration loaded successfully")
    print(f"   - Model: {config_info['model_path'] or 'Pretrained'}")
    print(f"   - Device: {config_info['device']}")
    print(f"   - Classes: {len(config_info['class_names'])}")
    
    # Test with sample images if available
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            print(f"\n🔍 Testing with single image: {input_path}")
            detect_single(pipeline, args)
        elif input_path.is_dir():
            print(f"\n🔍 Testing with directory: {input_path}")
            # Process only first few images for testing
            from src.preprocessing.image_utils import get_image_files
            image_files = get_image_files(input_path)[:3]  # Test with first 3 images
            
            for image_file in image_files:
                print(f"\nTesting: {image_file.name}")
                test_args = argparse.Namespace(
                    input=str(image_file),
                    output=args.output,
                    visualize=args.visualize
                )
                detect_single(pipeline, test_args)
    else:
        print("\n💡 To test with images, use: python main.py test --input <path>")
    
    print("\n✅ System test completed!")


if __name__ == "__main__":
    print("🔧 Steel Defect Detection System")
    print("================================")
    main()
