#!/usr/bin/env python3
"""
Test Steel Defect Detection Model
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration
MODEL_PATH = "models/weights/steel_defect_best.pt"
TEST_IMAGES_DIR = "data/labeled/validation/images"
OUTPUT_DIR = "results/test_predictions"

# Class names for our 6 defect types
CLASS_NAMES = {
    0: "crazing",
    1: "inclusion", 
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}

# Colors for visualization (BGR format for OpenCV)
COLORS = {
    0: (255, 0, 0),      # Blue - crazing
    1: (0, 255, 0),      # Green - inclusion
    2: (0, 0, 255),      # Red - patches
    3: (255, 255, 0),    # Cyan - pitted_surface
    4: (255, 0, 255),    # Magenta - rolled-in_scale
    5: (0, 255, 255)     # Yellow - scratches
}

def test_model():
    """Test the trained model on validation images."""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    # Load the trained model
    print(f"🤖 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get test images
    test_images_path = Path(TEST_IMAGES_DIR)
    if not test_images_path.exists():
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        return
    
    # Get sample images from each defect type
    image_files = []
    for defect_type in CLASS_NAMES.values():
        pattern_files = list(test_images_path.glob(f"{defect_type}*.jpg"))
        if pattern_files:
            # Take first 2 images of each type
            image_files.extend(pattern_files[:2])
    
    if not image_files:
        # If no pattern matches, just take first 10 images
        image_files = list(test_images_path.glob("*.jpg"))[:10]
    
    print(f"📸 Testing on {len(image_files)} images...")
    
    # Test each image
    results_summary = []
    
    for i, img_path in enumerate(image_files):
        print(f"\n🔍 Testing image {i+1}/{len(image_files)}: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=0.25, iou=0.45)
        
        # Load original image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id]
                })
                
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[class_id], 2)
                
                # Draw label
                label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), COLORS[class_id], -1)
                cv2.putText(img, label, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save result
        output_path = os.path.join(OUTPUT_DIR, f"result_{img_path.stem}.jpg")
        cv2.imwrite(output_path, img)
        
        # Print results
        if detections:
            print(f"   ✅ Found {len(detections)} defect(s):")
            for det in detections:
                print(f"      - {det['class_name']}: {det['confidence']:.3f}")
        else:
            print(f"   ℹ️  No defects detected")
        
        results_summary.append({
            'image': img_path.name,
            'detections': len(detections),
            'defects': [det['class_name'] for det in detections]
        })
    
    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print(f"=" * 50)
    print(f"Total images tested: {len(image_files)}")
    
    total_detections = sum(r['detections'] for r in results_summary)
    print(f"Total defects detected: {total_detections}")
    
    # Count detections by class
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    for result in results_summary:
        for defect in result['defects']:
            class_counts[defect] += 1
    
    print(f"\nDetections by class:")
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"  {class_name}: {count}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"🎉 Model testing completed!")

def quick_test():
    """Quick test on a few sample images."""
    print("🚀 Quick Model Test")
    print("=" * 30)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Get first 3 validation images
    test_images_path = Path(TEST_IMAGES_DIR)
    if not test_images_path.exists():
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        return
    
    image_files = list(test_images_path.glob("*.jpg"))[:3]
    
    if not image_files:
        print("❌ No test images found")
        return
    
    print(f"Testing {len(image_files)} sample images...")
    
    for i, img_path in enumerate(image_files):
        print(f"\n📸 Image {i+1}: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=0.25, verbose=False)
        
        # Count detections
        detection_count = 0
        if len(results) > 0 and results[0].boxes is not None:
            detection_count = len(results[0].boxes)
            
            # Print detected classes
            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = box.conf[0].cpu().numpy()
                print(f"   ✅ {CLASS_NAMES[class_id]}: {confidence:.3f}")
        
        if detection_count == 0:
            print(f"   ℹ️  No defects detected")

if __name__ == "__main__":
    print("🔬 STEEL DEFECT DETECTION MODEL TEST")
    print("=" * 40)
    
    # Run quick test first
    quick_test()
    
    print("\n" + "=" * 40)
    choice = input("Run full test with image outputs? (y/n): ").strip().lower()
    
    if choice == 'y':
        test_model()
    else:
        print("✅ Quick test completed!")
