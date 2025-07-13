#!/usr/bin/env python3
"""
Flask Web Application for Steel Defect Detection Dashboard
Integrates with YOLOv8 model and provides real-time detection via WebSocket
"""

import os
import sys
import cv2
import json
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import logging
from collections import defaultdict, deque
import base64

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.detection.yolo_detector import SteelDefectDetector
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='dashboard/templates',
            static_folder='frontend/static')
app.config['SECRET_KEY'] = 'steel-defect-detection-secret-key'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class DetectionDashboard:
    """Main dashboard class handling camera, detection, and WebSocket communication"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config = Config(config_path)
        
        # Initialize components
        model_path = self.config.get('model', {}).get('checkpoint_path', 'runs/detect/train/weights/best.pt')
        config_dict = self.config.get_model_config()
        self.detector = SteelDefectDetector(model_path=model_path, config=config_dict)
        
        # Camera and detection state
        self.camera = None
        self.detection_active = False
        self.camera_active = False
        self.detection_thread = None
        self.camera_thread = None
        
        # Statistics tracking
        self.detection_stats = {
            'total_detections': 0,
            'defect_counts': defaultdict(int),
            'detection_history': deque(maxlen=100),
            'fps': 0,
            'processing_time': 0
        }
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained YOLOv8 model"""
        try:
            model_path = self.config.get('model', {}).get('checkpoint_path', 'runs/detect/train/weights/best.pt')
            if os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                self.model = YOLO(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {model_path}, using default YOLOv8n")
                self.model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = YOLO('yolov8n.pt')
    
    def initialize_camera(self, camera_id=0):
        """Initialize camera capture"""
        try:
            if self.camera:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(camera_id)
            
            # Set camera properties
            camera_config = self.config.get('camera', {})
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 640))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 480))
            self.camera.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
                
            logger.info(f"Camera {camera_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def start_camera_stream(self):
        """Start camera streaming thread"""
        if not self.camera_active:
            self.camera_active = True
            self.camera_thread = threading.Thread(target=self._camera_stream_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            logger.info("Camera stream started")
    
    def stop_camera_stream(self):
        """Stop camera streaming"""
        self.camera_active = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.camera:
            self.camera.release()
        logger.info("Camera stream stopped")
    
    def _camera_stream_loop(self):
        """Camera streaming loop"""
        while self.camera_active:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret:
                        # Convert frame to base64 for WebSocket transmission
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Emit frame to connected clients
                        socketio.emit('camera_frame', {
                            'frame': frame_b64,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Process detection if active
                        if self.detection_active:
                            self._process_detection(frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in camera stream: {e}")
                time.sleep(1)
    
    def _process_detection(self, frame):
        """Process detection on frame"""
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model(frame)
            
            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            'class_id': int(box.cls[0]),
                            'class_name': self.model.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        }
                        detections.append(detection)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_statistics(detections, processing_time)
            
            # Create annotated frame
            annotated_frame = self._draw_detections(frame, detections)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit detection results
            socketio.emit('detection_results', {
                'frame': frame_b64,
                'detections': detections,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in detection processing: {e}")
    
    def _update_statistics(self, detections, processing_time):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(detections)
        self.detection_stats['processing_time'] = processing_time
        self.detection_stats['fps'] = 1.0 / processing_time if processing_time > 0 else 0
        
        # Update defect counts
        for detection in detections:
            self.detection_stats['defect_counts'][detection['class_name']] += 1
        
        # Add to history
        self.detection_stats['detection_history'].append({
            'timestamp': datetime.now().isoformat(),
            'count': len(detections),
            'processing_time': processing_time
        })
    
    def start_detection(self):
        """Start detection process"""
        self.detection_active = True
        logger.info("Detection started")
        
        # Emit status update
        socketio.emit('detection_status', {
            'active': True,
            'message': 'Detection started'
        })
    
    def stop_detection(self):
        """Stop detection process"""
        self.detection_active = False
        logger.info("Detection stopped")
        
        # Emit status update
        socketio.emit('detection_status', {
            'active': False,
            'message': 'Detection stopped'
        })
    
    def _draw_detections(self, frame, detections):
        """Draw detection boxes on frame"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'total_detections': self.detection_stats['total_detections'],
            'defect_counts': dict(self.detection_stats['defect_counts']),
            'fps': self.detection_stats['fps'],
            'processing_time': self.detection_stats['processing_time'],
            'history': list(self.detection_stats['detection_history'])
        }

# Initialize dashboard
dashboard = DetectionDashboard()

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'camera_active': dashboard.camera_active,
        'detection_active': dashboard.detection_active,
        'model_loaded': dashboard.model is not None,
        'statistics': dashboard.get_statistics()
    })

@app.route('/api/statistics')
def get_statistics():
    """Get detection statistics"""
    return jsonify(dashboard.get_statistics())

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to steel defect detection dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_camera')
def handle_start_camera(data):
    """Handle start camera command"""
    camera_id = data.get('camera_id', 0)
    
    if dashboard.initialize_camera(camera_id):
        dashboard.start_camera_stream()
        emit('camera_status', {
            'active': True,
            'message': f'Camera {camera_id} started successfully'
        })
    else:
        emit('camera_status', {
            'active': False,
            'message': f'Failed to start camera {camera_id}'
        })

@socketio.on('stop_camera')
def handle_stop_camera():
    """Handle stop camera command"""
    dashboard.stop_camera_stream()
    emit('camera_status', {
        'active': False,
        'message': 'Camera stopped'
    })

@socketio.on('start_detection')
def handle_start_detection():
    """Handle start detection command"""
    dashboard.start_detection()

@socketio.on('stop_detection')
def handle_stop_detection():
    """Handle stop detection command"""
    dashboard.stop_detection()

@socketio.on('get_statistics')
def handle_get_statistics():
    """Handle get statistics command"""
    emit('statistics_update', dashboard.get_statistics())

@socketio.on('reset_statistics')
def handle_reset_statistics():
    """Handle reset statistics command"""
    dashboard.detection_stats = {
        'total_detections': 0,
        'defect_counts': defaultdict(int),
        'detection_history': deque(maxlen=100),
        'fps': 0,
        'processing_time': 0
    }
    emit('statistics_reset', {'message': 'Statistics reset successfully'})

if __name__ == '__main__':
    try:
        logger.info("Starting Steel Defect Detection Dashboard...")
        logger.info("Dashboard will be available at http://localhost:5000")
        
        # Run the Flask app with SocketIO
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=True, 
                    use_reloader=False)
                    
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")
        dashboard.stop_camera_stream()
        dashboard.stop_detection()
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)
