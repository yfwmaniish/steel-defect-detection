"""
Configuration utilities for Steel Defect Detection System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the steel defect detection system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_defect_classes(self) -> Dict[int, str]:
        """Get defect class mapping."""
        return self.get('defects.classes', {})
    
    def get_defect_colors(self) -> Dict[int, list]:
        """Get defect color mapping for visualization."""
        return self.get('defects.colors', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self.get('cameras', {})
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        paths = [
            self.get('data.raw_images_path'),
            self.get('data.labeled_data_path'),
            self.get('data.processed_images_path'),
            self.get('data.results_path'),
            self.get('model.weights_path'),
            'logs'
        ]
        
        for path in paths:
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {path}")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def setup_logging(config: Config):
    """Setup logging configuration."""
    import logging
    
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('logging.file', 'logs/steel_defect.log')
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


if __name__ == "__main__":
    # Test configuration loading
    config = Config()
    print("Configuration loaded successfully!")
    print(f"Defect classes: {config.get_defect_classes()}")
    print(f"Model version: {config.get('model.version')}")
    
    # Create directories
    config.create_directories()
