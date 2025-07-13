"""
Detection Pipeline for Steel Defect Detection System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
import json
import time
from tqdm import tqdm

from src.utils.config import Config, setup_logging
from src.preprocessing.image_utils import get_image_files, parse_image_name, ImagePreprocessor
from src.detection.yolo_detector import SteelDefectDetector

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Complete detection pipeline for steel defect detection."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize detection pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.config.create_directories()
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(
            target_size=self.config.get('data.image_size', [720, 720])
        )
        
        # Initialize detector
        model_config = self.config.get_model_config()
        weights_path = self.config.get('model.weights_path')
        
        # Look for trained weights
        trained_weights = None
        if weights_path:
            weights_dir = Path(weights_path)
            if weights_dir.exists():
                weight_files = list(weights_dir.glob("*.pt"))
                if weight_files:
                    trained_weights = str(weight_files[0])  # Use first found weights
        
        self.detector = SteelDefectDetector(
            model_path=trained_weights,
            config=model_config
        )
        
        # Initialize results storage
        self.results = []
        
        logger.info("Detection pipeline initialized successfully")
    
    def process_single_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Process a single image for defect detection.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processing result dictionary
        """
        image_path = Path(image_path)
        start_time = time.time()
        
        try:
            # Parse image information
            image_id, camera_position = parse_image_name(image_path)
            
            # Load and preprocess image
            image = self.preprocessor.preprocess_image(image_path)
            if image is None:
                logger.error(f"Failed to preprocess image: {image_path}")
                return self._create_error_result(image_path, "preprocessing_failed")
            
            # Detect defects
            detections = self.detector.detect(image)
            
            # Process detection results
            processing_time = time.time() - start_time
            result = {
                'image_path': str(image_path),
                'image_id': image_id,
                'camera_position': camera_position,
                'processing_time': processing_time,
                'num_defects': len(detections),
                'detections': detections,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Save individual detections to results
            for detection in detections:
                detection_record = {
                    'timestamp': result['timestamp'],
                    'image_id': image_id,
                    'camera': camera_position,
                    'defect_type': detection['class_name'],
                    'x': detection['bbox']['x'],
                    'y': detection['bbox']['y'],
                    'width': detection['bbox']['width'],
                    'height': detection['bbox']['height'],
                    'confidence': detection['confidence'],
                    'area': detection['area']
                }
                self.results.append(detection_record)
            
            logger.info(f"Processed {image_path.name}: {len(detections)} defects in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return self._create_error_result(image_path, str(e))
    
    def process_batch(self, input_dir: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     save_visualizations: bool = False) -> List[Dict]:
        """
        Process a batch of images for defect detection.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results (optional)
            save_visualizations: Whether to save detection visualizations
            
        Returns:
            List of processing results
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return []
        
        # Get all image files
        image_files = get_image_files(input_dir)
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []
        
        logger.info(f"Processing {len(image_files)} images from {input_dir}")
        
        # Process images with progress bar
        batch_results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single_image(image_path)
            batch_results.append(result)
            
            # Save visualization if requested
            if save_visualizations and result['success'] and result['num_defects'] > 0:
                self._save_visualization(image_path, result, output_dir)
        
        # Save batch results
        if output_dir:
            self._save_batch_results(batch_results, output_dir)
        
        logger.info(f"Batch processing completed. Total images: {len(image_files)}")
        return batch_results
    
    def process_camera_pair(self, image_id: str, input_dir: Union[str, Path],
                           output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Process a pair of images (top and bottom camera) for the same steel sheet.
        
        Args:
            image_id: ID of the steel sheet
            input_dir: Directory containing input images
            output_dir: Directory to save results (optional)
            
        Returns:
            Combined processing result
        """
        input_dir = Path(input_dir)
        
        # Look for top and bottom camera images
        top_image = None
        bottom_image = None
        
        for suffix in ['_up', '_top']:
            candidate = input_dir / f"{image_id}{suffix}.jpg"
            if candidate.exists():
                top_image = candidate
                break
        
        for suffix in ['_down', '_bottom']:
            candidate = input_dir / f"{image_id}{suffix}.jpg"
            if candidate.exists():
                bottom_image = candidate
                break
        
        results = {}
        
        # Process top camera image
        if top_image:
            results['top'] = self.process_single_image(top_image)
        else:
            logger.warning(f"Top camera image not found for ID: {image_id}")
            results['top'] = None
        
        # Process bottom camera image
        if bottom_image:
            results['bottom'] = self.process_single_image(bottom_image)
        else:
            logger.warning(f"Bottom camera image not found for ID: {image_id}")
            results['bottom'] = None
        
        # Combine results
        combined_result = {
            'image_id': image_id,
            'top_result': results['top'],
            'bottom_result': results['bottom'],
            'total_defects': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Count total defects
        for camera_result in results.values():
            if camera_result and camera_result['success']:
                combined_result['total_defects'] += camera_result['num_defects']
        
        logger.info(f"Camera pair processing completed for {image_id}: {combined_result['total_defects']} total defects")
        return combined_result
    
    def save_results_to_csv(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Save detection results to CSV file.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            Path to saved CSV file
        """
        if output_path is None:
            output_path = self.config.get('output.csv_path', 'results/detections.csv')
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame from results
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to CSV: {output_path}")
        else:
            logger.warning("No results to save")
        
        return str(output_path)
    
    def get_detection_summary(self) -> Dict:
        """
        Get summary of detection results.
        
        Returns:
            Summary statistics
        """
        if not self.results:
            return {"total_detections": 0, "defect_types": {}}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_detections': len(df),
            'defect_types': df['defect_type'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'cameras_processed': df['camera'].nunique(),
            'images_processed': df['image_id'].nunique()
        }
        
        return summary
    
    def _create_error_result(self, image_path: Path, error_message: str) -> Dict:
        """Create error result dictionary."""
        return {
            'image_path': str(image_path),
            'image_id': image_path.stem,
            'camera_position': 'unknown',
            'processing_time': 0,
            'num_defects': 0,
            'detections': [],
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error_message
        }
    
    def _save_visualization(self, image_path: Path, result: Dict, output_dir: Optional[Path]):
        """Save detection visualization."""
        if output_dir is None:
            output_dir = Path("results/visualizations")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image for visualization
        image = self.preprocessor.load_image(image_path)
        if image is not None:
            vis_path = output_dir / f"{image_path.stem}_detected.jpg"
            self.detector.visualize_detections(image, result['detections'], str(vis_path))
    
    def _save_batch_results(self, batch_results: List[Dict], output_dir: Path):
        """Save batch processing results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_file = output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        logger.info(f"Batch results saved to: {results_file}")


if __name__ == "__main__":
    # Test detection pipeline
    pipeline = DetectionPipeline()
    
    # Test with sample images
    input_dir = "data/raw"
    if Path(input_dir).exists():
        results = pipeline.process_batch(input_dir, save_visualizations=True)
        
        # Save results
        csv_path = pipeline.save_results_to_csv()
        summary = pipeline.get_detection_summary()
        
        print(f"Processing completed!")
        print(f"Results saved to: {csv_path}")
        print(f"Summary: {summary}")
    else:
        print(f"Input directory not found: {input_dir}")
        print("Place test images in data/raw/ directory to test the pipeline")
