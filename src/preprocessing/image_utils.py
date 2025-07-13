"""
Image preprocessing utilities for Steel Defect Detection System
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing utilities for steel defect detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (720, 720)):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            if image_path.suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported image format: {image_path.suffix}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def resize_image(self, image: np.ndarray, maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect_ratio:
            return self._resize_with_padding(image)
        else:
            return cv2.resize(image, self.target_size)
    
    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio and adding padding.
        
        Args:
            image: Input image
            
        Returns:
            Resized image with padding
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image on canvas
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def enhance_contrast(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image
            clip_limit: Clipping limit for CLAHE
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def reduce_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Reduce image noise.
        
        Args:
            image: Input image
            method: Noise reduction method ('gaussian', 'bilateral', 'median')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            return image
    
    def preprocess_image(self, image_path: Union[str, Path], 
                        enhance_contrast: bool = True,
                        reduce_noise: bool = True) -> Optional[np.ndarray]:
        """
        Complete image preprocessing pipeline.
        
        Args:
            image_path: Path to image file
            enhance_contrast: Whether to enhance contrast
            reduce_noise: Whether to reduce noise
            
        Returns:
            Preprocessed image or None if failed
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Resize image
        image = self.resize_image(image)
        
        # Enhance contrast
        if enhance_contrast:
            image = self.enhance_contrast(image)
        
        # Reduce noise
        if reduce_noise:
            image = self.reduce_noise(image)
        
        return image
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            success = cv2.imwrite(str(output_path), image_bgr)
            if success:
                logger.info(f"Image saved successfully: {output_path}")
                return True
            else:
                logger.error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving image {output_path}: {str(e)}")
            return False


def get_image_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return []
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    if recursive:
        for ext in supported_formats:
            image_files.extend(directory.rglob(f"*{ext}"))
            image_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in supported_formats:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def parse_image_name(image_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Parse image name to extract ID and camera position.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (image_id, camera_position)
    """
    image_path = Path(image_path)
    filename = image_path.stem
    
    # Parse filename like "01_up" or "01_down"
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 2:
            image_id = parts[0]
            camera_suffix = parts[1]
            
            # Map suffix to camera position
            if camera_suffix.lower() in ['up', 'top']:
                camera_position = 'top'
            elif camera_suffix.lower() in ['down', 'bottom']:
                camera_position = 'bottom'
            else:
                camera_position = camera_suffix
                
            return image_id, camera_position
    
    # If parsing fails, return filename as ID
    return filename, 'unknown'


if __name__ == "__main__":
    # Test image preprocessing
    preprocessor = ImagePreprocessor()
    
    # Test with a sample image path
    sample_path = "data/raw/01_up.jpg"
    if Path(sample_path).exists():
        processed = preprocessor.preprocess_image(sample_path)
        if processed is not None:
            print(f"Successfully preprocessed image: {sample_path}")
            print(f"Processed image shape: {processed.shape}")
        else:
            print(f"Failed to preprocess image: {sample_path}")
    else:
        print(f"Sample image not found: {sample_path}")
        print("Place test images in data/raw/ directory to test preprocessing")
