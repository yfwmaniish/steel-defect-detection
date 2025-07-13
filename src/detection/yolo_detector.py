"""
YOLOv8 Detection Module for Steel Defect Detection System
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from ultralytics import YOLO
import cv2
from datetime import datetime

logger = logging.getLogger(__name__)


class SteelDefectDetector:
    """YOLOv8-based steel defect detector."""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the steel defect detector.
        
        Args:
            model_path: Path to trained model weights
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_path = model_path
        self.model = None
        self.class_names = self._get_class_names()
        self.device = self._get_device()
        
        # Detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.max_detections = self.config.get('max_detections', 1000)
        
        # Load model
        self._load_model()
        
        logger.info(f"Steel defect detector initialized with device: {self.device}")
    
    def _get_class_names(self) -> Dict[int, str]:
        """Get defect class names from config."""
        default_classes = {
            0: "scratch",
            1: "dent", 
            2: "bend",
            3: "color_defect",
            4: "hole",
            5: "patch"
        }
        return self.config.get('classes', default_classes)
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            return device_config
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            if self.model_path and Path(self.model_path).exists():
                # Load trained model
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded trained model from: {self.model_path}")
            else:
                # Load pretrained model
                model_version = self.config.get('version', 'yolov8n')
                self.model = YOLO(f"{model_version}.pt")
                logger.info(f"Loaded pretrained model: {model_version}")
            
            # Move model to device
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def detect(self, image: np.ndarray, save_results: bool = False) -> List[Dict]:
        """
        Detect defects in a single image.
        
        Args:
            image: Input image as numpy array
            save_results: Whether to save detection results
            
        Returns:
            List of detection results
        """
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False
            )
            
            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        detection = {
                            'class_id': int(cls_id),
                            'class_name': self.class_names.get(int(cls_id), 'unknown'),
                            'confidence': float(conf),
                            'bbox': {
                                'x': float(x1),
                                'y': float(y1),
                                'width': float(width),
                                'height': float(height)
                            },
                            'area': float(area),
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} defects")
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect defects in multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of detection results for each image
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            detections = self.detect(image)
            results.append(detections)
        
        return results
    
    def detect_from_path(self, image_path: Union[str, Path]) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Detect defects from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (detection_results, loaded_image)
        """
        try:
            image_path = Path(image_path)
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return [], None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect defects
            detections = self.detect(image_rgb)
            
            return detections, image_rgb
            
        except Exception as e:
            logger.error(f"Error detecting from path {image_path}: {str(e)}")
            return [], None
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Input image
            detections: Detection results
            save_path: Optional path to save visualization
            
        Returns:
            Image with detection visualizations
        """
        vis_image = image.copy()
        
        # Define colors for each class
        colors = {
            0: (255, 0, 0),    # Red - scratch
            1: (0, 255, 0),    # Green - dent
            2: (0, 0, 255),    # Blue - bend
            3: (255, 255, 0),  # Yellow - color_defect
            4: (255, 0, 255),  # Magenta - hole
            5: (0, 255, 255)   # Cyan - patch
        }
        
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get coordinates
            x1 = int(bbox['x'])
            y1 = int(bbox['y'])
            x2 = int(x1 + bbox['width'])
            y2 = int(y1 + bbox['height'])
            
            # Get color
            color = colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save visualization if requested
        if save_path:
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, vis_image_bgr)
            logger.info(f"Visualization saved to: {save_path}")
        
        return vis_image
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections
        }
    
    def update_thresholds(self, confidence: Optional[float] = None, 
                         iou: Optional[float] = None):
        """Update detection thresholds."""
        if confidence is not None:
            self.confidence_threshold = confidence
            logger.info(f"Updated confidence threshold to: {confidence}")
        
        if iou is not None:
            self.iou_threshold = iou
            logger.info(f"Updated IoU threshold to: {iou}")


if __name__ == "__main__":
    # Test the detector
    config = {
        'version': 'yolov8n',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'classes': {
            0: "scratch",
            1: "dent",
            2: "bend", 
            3: "color_defect",
            4: "hole",
            5: "patch"
        }
    }
    
    detector = SteelDefectDetector(config=config)
    print("Steel defect detector initialized successfully!")
    print(f"Model info: {detector.get_model_info()}")
    
    # Test with sample image if available
    sample_image_path = "data/raw/01_up.jpg"
    if Path(sample_image_path).exists():
        detections, image = detector.detect_from_path(sample_image_path)
        print(f"Found {len(detections)} defects in sample image")
        
        if detections and image is not None:
            # Visualize results
            vis_image = detector.visualize_detections(image, detections)
            print("Detection visualization created")
    else:
        print(f"Sample image not found: {sample_image_path}")
        print("Place test images in data/raw/ directory to test detection")
