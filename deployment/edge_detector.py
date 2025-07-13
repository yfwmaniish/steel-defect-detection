#!/usr/bin/env python3
"""
Steel Defect Detection - Edge Device Implementation
Real-time defect detection system for edge deployment
"""

import cv2
import time
import json
import sqlite3
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
import yaml
import os
import signal
import sys

# Hardware control imports (install with: pip install RPi.GPIO gpiozero)
try:
    import RPi.GPIO as GPIO
    from gpiozero import LED, Button, Buzzer
    RASPBERRY_PI = True
except ImportError:
    RASPBERRY_PI = False
    print("⚠️  RPi.GPIO not available - running in simulation mode")

class EdgeDefectDetector:
    """
    Edge device implementation for real-time steel defect detection
    """
    
    def __init__(self, config_path: str = "config/edge_config.yaml"):
        """Initialize the edge detection system"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_database()
        self.setup_hardware()
        self.load_ai_model()
        
        # System state
        self.running = False
        self.last_inspection = None
        self.total_inspections = 0
        self.defects_today = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def load_config(self, config_path: str) -> dict:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'camera': {
                    'device_id': 0,
                    'resolution': [1920, 1080],
                    'fps': 30
                },
                'ai_model': {
                    'model_path': 'models/steel_defect_advanced.pt',
                    'confidence_threshold': 0.5,
                    'input_size': 640
                },
                'hardware': {
                    'status_led_pin': 18,
                    'error_led_pin': 19,
                    'buzzer_pin': 20,
                    'trigger_button_pin': 21
                },
                'alerts': {
                    'visual': True,
                    'audio': True
                },
                'logging': {
                    'level': 'INFO',
                    'file': 'logs/edge_detector.log'
                }
            }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Edge Defect Detector initialized")
    
    def setup_database(self):
        """Setup local SQLite database"""
        db_dir = Path("data")
        db_dir.mkdir(exist_ok=True)
        
        self.db_path = db_dir / "inspections.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    total_defects INTEGER DEFAULT 0,
                    pass_fail TEXT CHECK(pass_fail IN ('PASS', 'FAIL')),
                    processing_time REAL,
                    confidence_avg REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS defects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id INTEGER,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL,
                    bbox_y1 REAL,
                    bbox_x2 REAL,
                    bbox_y2 REAL,
                    FOREIGN KEY (inspection_id) REFERENCES inspections (id)
                )
            """)
            
            conn.commit()
    
    def setup_hardware(self):
        """Setup hardware components (LEDs, buttons, etc.)"""
        if not RASPBERRY_PI:
            self.logger.warning("Running in simulation mode - no hardware control")
            return
        
        try:
            # Status indicators
            self.status_led = LED(self.config['hardware']['status_led_pin'])
            self.error_led = LED(self.config['hardware']['error_led_pin'])
            self.buzzer = Buzzer(self.config['hardware']['buzzer_pin'])
            
            # Input controls
            self.trigger_button = Button(self.config['hardware']['trigger_button_pin'])
            self.trigger_button.when_pressed = self.manual_inspection
            
            # Initial state
            self.status_led.on()  # System ready
            self.error_led.off()
            
            self.logger.info("✅ Hardware components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Hardware setup failed: {e}")
    
    def load_ai_model(self):
        """Load the YOLO model for inference"""
        try:
            model_path = self.config['ai_model']['model_path']
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.logger.info(f"✅ AI model loaded: {model_path}")
            
            # Define class names
            self.class_names = {
                0: 'crazing',
                1: 'inclusion',
                2: 'patches', 
                3: 'pitted_surface',
                4: 'rolled-in_scale',
                5: 'scratches'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load AI model: {e}")
            raise
    
    def initialize_camera(self):
        """Initialize camera for image capture"""
        try:
            self.camera = cv2.VideoCapture(self.config['camera']['device_id'])
            
            # Set camera properties
            width, height = self.config['camera']['resolution']
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
            
            self.logger.info("✅ Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Camera initialization failed: {e}")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture image from camera"""
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.error("Failed to capture image")
                return None
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Image capture failed: {e}")
            return None
    
    def detect_defects(self, image: np.ndarray) -> Dict:
        """Run defect detection on captured image"""
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.config['ai_model']['confidence_threshold'])
            result = results[0]
            
            # Extract detections
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls.item()),
                        'class_name': self.class_names[int(box.cls.item())],
                        'confidence': float(box.conf.item()),
                        'bbox': box.xyxy.cpu().numpy().tolist()[0]
                    }
                    detections.append(detection)
            
            processing_time = time.time() - start_time
            
            # Calculate results
            total_defects = len(detections)
            confidence_avg = np.mean([d['confidence'] for d in detections]) if detections else 0.0
            pass_fail = "PASS" if total_defects == 0 else "FAIL"
            
            inspection_result = {
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'total_defects': total_defects,
                'confidence_avg': float(confidence_avg),
                'processing_time': processing_time,
                'pass_fail': pass_fail,
                'annotated_image': result.plot() if result.boxes is not None else image
            }
            
            self.logger.info(f"🔍 Inspection complete: {total_defects} defects found ({processing_time:.2f}s)")
            return inspection_result
            
        except Exception as e:
            self.logger.error(f"❌ Detection failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'detections': [],
                'total_defects': 0,
                'confidence_avg': 0.0,
                'processing_time': 0.0,
                'pass_fail': 'ERROR',
                'annotated_image': image,
                'error': str(e)
            }
    
    def save_inspection_result(self, image: np.ndarray, result: Dict) -> int:
        """Save inspection result to database and files"""
        try:
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_dir = Path("data/images")
            image_dir.mkdir(exist_ok=True)
            
            image_path = image_dir / f"inspection_{timestamp}.jpg"
            annotated_path = image_dir / f"annotated_{timestamp}.jpg"
            
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(annotated_path), result['annotated_image'])
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO inspections 
                    (image_path, total_defects, pass_fail, processing_time, confidence_avg)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    str(image_path),
                    result['total_defects'],
                    result['pass_fail'],
                    result['processing_time'],
                    result['confidence_avg']
                ))
                
                inspection_id = cursor.lastrowid
                
                # Save individual defects
                for detection in result['detections']:
                    conn.execute("""
                        INSERT INTO defects 
                        (inspection_id, defect_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        inspection_id,
                        detection['class_name'],
                        detection['confidence'],
                        detection['bbox'][0],
                        detection['bbox'][1],
                        detection['bbox'][2],
                        detection['bbox'][3]
                    ))
                
                conn.commit()
            
            self.logger.info(f"💾 Inspection result saved (ID: {inspection_id})")
            return inspection_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save inspection result: {e}")
            return -1
    
    def trigger_alerts(self, result: Dict):
        """Trigger visual and audio alerts based on inspection result"""
        if not RASPBERRY_PI:
            return
        
        try:
            if result['pass_fail'] == 'FAIL':
                # Defects found - trigger alerts
                if self.config['alerts']['visual']:
                    self.error_led.blink(on_time=0.5, off_time=0.5, n=5)
                
                if self.config['alerts']['audio']:
                    self.buzzer.beep(on_time=0.1, off_time=0.1, n=3)
                
                self.logger.warning(f"🚨 DEFECTS DETECTED: {result['total_defects']} defects found")
                
            elif result['pass_fail'] == 'PASS':
                # No defects - brief positive indication
                if self.config['alerts']['visual']:
                    self.status_led.blink(on_time=0.2, off_time=0.2, n=2)
                
                self.logger.info("✅ INSPECTION PASSED: No defects detected")
                
        except Exception as e:
            self.logger.error(f"Alert system error: {e}")
    
    def manual_inspection(self):
        """Trigger manual inspection via button press"""
        if not self.running:
            return
        
        self.logger.info("🔘 Manual inspection triggered")
        threading.Thread(target=self.run_single_inspection, daemon=True).start()
    
    def run_single_inspection(self):
        """Run a single inspection cycle"""
        try:
            # Capture image
            image = self.capture_image()
            if image is None:
                return
            
            # Detect defects
            result = self.detect_defects(image)
            
            # Save results
            self.save_inspection_result(image, result)
            
            # Trigger alerts
            self.trigger_alerts(result)
            
            # Update statistics
            self.total_inspections += 1
            if result['pass_fail'] == 'FAIL':
                self.defects_today += result['total_defects']
            
            self.last_inspection = result
            
        except Exception as e:
            self.logger.error(f"❌ Inspection failed: {e}")
    
    def continuous_monitoring(self):
        """Run continuous monitoring mode"""
        self.logger.info("🔄 Starting continuous monitoring...")
        
        while self.running:
            try:
                # Run inspection
                self.run_single_inspection()
                
                # Wait between inspections
                time.sleep(5)  # 5 second interval
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(10)  # Longer wait on error
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'status': 'running' if self.running else 'stopped',
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            'total_inspections': self.total_inspections,
            'defects_today': self.defects_today,
            'last_inspection': self.last_inspection,
            'camera_connected': hasattr(self, 'camera') and self.camera.isOpened(),
            'model_loaded': hasattr(self, 'model')
        }
    
    def start(self):
        """Start the edge detection system"""
        try:
            self.logger.info("🚀 Starting Steel Defect Detection System...")
            
            # Initialize camera
            if not self.initialize_camera():
                return False
            
            self.running = True
            self.start_time = time.time()
            
            # Start continuous monitoring in separate thread
            monitoring_thread = threading.Thread(target=self.continuous_monitoring, daemon=True)
            monitoring_thread.start()
            
            self.logger.info("✅ System started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start system: {e}")
            return False
    
    def stop(self):
        """Stop the edge detection system"""
        self.logger.info("🛑 Stopping Steel Defect Detection System...")
        
        self.running = False
        
        # Cleanup camera
        if hasattr(self, 'camera'):
            self.camera.release()
        
        # Cleanup GPIO
        if RASPBERRY_PI:
            try:
                self.status_led.off()
                self.error_led.off()
                GPIO.cleanup()
            except:
                pass
        
        self.logger.info("✅ System stopped")
    
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.logger.info(f"📶 Received signal {signum}")
        self.stop()
        sys.exit(0)


def main():
    """Main entry point"""
    print("🔍 Steel Defect Detection - Edge Device")
    print("="*50)
    
    # Create detector instance
    detector = EdgeDefectDetector()
    
    try:
        # Start the system
        if detector.start():
            print("✅ System running... Press Ctrl+C to stop")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
        else:
            print("❌ Failed to start system")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
