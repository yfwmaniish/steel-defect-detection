"""
Convert NEU Dataset from XML to YOLO Format
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

# Configuration - Map NEU-DET defects to unique classes
CLASSES = {
    'crazing': 0,        # crazing
    'inclusion': 1,      # inclusion
    'patches': 2,        # patches
    'pitted_surface': 3, # pitted_surface
    'rolled-in_scale': 4, # rolled-in_scale
    'scratches': 5       # scratches
}

DATASET_PATH = Path("data/downloaded/NEU-DET")
OUTPUT_PATH = Path("data/labeled")


def convert_annotation(xml_path, txt_path):
    """
    Convert XML annotations to YOLO txt format.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(txt_path, 'w') as txt_file:
        for obj in root.findall('object'):
            name = obj.find('name').text
            cls_id = CLASSES.get(name)
            if cls_id is None:
                continue
            
            xml_box = obj.find('bndbox')
            b = (
                int(xml_box.find('xmin').text),
                int(xml_box.find('xmax').text),
                int(xml_box.find('ymin').text),
                int(xml_box.find('ymax').text)
            )
            
            # YOLO format: cls_id x_center y_center width height (normalized)
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            
            x_center = (b[0] + b[1]) / 2.0 / width
            y_center = (b[2] + b[3]) / 2.0 / height
            bbox_width = (b[1] - b[0]) / width
            bbox_height = (b[3] - b[2]) / height
            
            txt_file.write(f"{cls_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")


def process_dataset():
    """
    Process and convert entire NEU dataset.
    """
    if not DATASET_PATH.exists():
        print(f"Dataset path not found: {DATASET_PATH}")
        return

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Setup directories
    for split in ['train', 'validation']:
        for folder in ['images', 'labels']:
            Path(OUTPUT_PATH / split / folder).mkdir(parents=True, exist_ok=True)

    # Process each set
    for split in ['train', 'validation']:
        anno_dir = DATASET_PATH / split / 'annotations'
        img_base_dir = DATASET_PATH / split / 'images'
        output_img_dir = OUTPUT_PATH / split / 'images'
        output_lbl_dir = OUTPUT_PATH / split / 'labels'

        # Process each defect type directory
        for defect_type in CLASSES.keys():
            defect_img_dir = img_base_dir / defect_type
            if defect_img_dir.exists():
                # Process all images in this defect type directory
                for img_file in defect_img_dir.glob('*.jpg'):
                    base_name = img_file.stem
                    xml_file = anno_dir / f"{base_name}.xml"
                    
                    if xml_file.exists():
                        # Convert annotation
                        convert_annotation(xml_file, output_lbl_dir / f"{base_name}.txt")
                        
                        # Copy image
                        shutil.copy(img_file, output_img_dir / img_file.name)
        
    print(f"Conversion completed. YOLO formatted data available at: {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Starting conversion of NEU dataset to YOLO format...")
    process_dataset()
