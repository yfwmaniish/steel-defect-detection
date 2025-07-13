"""
Dataset Structure Analyzer
"""

import os
from pathlib import Path
import json

def analyze_dataset_structure(dataset_path):
    """Analyze and display dataset structure."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    print(f"📁 Analyzing dataset: {dataset_path}")
    print("=" * 50)
    
    # Count files by extension
    file_counts = {}
    total_files = 0
    
    # Walk through directory
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Show first few files in each directory
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files[:5]):  # Show first 5 files
            print(f"{subindent}{file}")
            
            # Count extensions
            ext = Path(file).suffix.lower()
            file_counts[ext] = file_counts.get(ext, 0) + 1
            total_files += 1
        
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    print("\n📊 File Summary:")
    print(f"Total files: {total_files}")
    for ext, count in file_counts.items():
        print(f"  {ext or 'no extension'}: {count} files")
    
    # Look for common dataset patterns
    print("\n🔍 Dataset Analysis:")
    
    # Check for class folders
    class_folders = []
    for item in dataset_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            class_folders.append(item.name)
    
    if class_folders:
        print(f"📋 Found {len(class_folders)} potential class folders:")
        for folder in class_folders:
            folder_path = dataset_path / folder
            image_count = len([f for f in folder_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            print(f"  - {folder}: {image_count} images")
    
    # Check for labels
    label_files = list(dataset_path.rglob('*.txt'))
    if label_files:
        print(f"📄 Found {len(label_files)} label files (.txt)")
    
    # Check for YOLO format
    yaml_files = list(dataset_path.rglob('*.yaml')) + list(dataset_path.rglob('*.yml'))
    if yaml_files:
        print(f"📝 Found {len(yaml_files)} YAML files")
    
    return {
        'total_files': total_files,
        'file_counts': file_counts,
        'class_folders': class_folders,
        'label_files': len(label_files),
        'yaml_files': len(yaml_files)
    }

if __name__ == "__main__":
    print("📊 Dataset Structure Analyzer")
    print("=" * 30)
    
    dataset_path = input("Enter path to your dataset folder: ").strip()
    
    if dataset_path:
        result = analyze_dataset_structure(dataset_path)
        
        # Save analysis
        with open("data/dataset_analysis.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Analysis saved to: data/dataset_analysis.json")
    else:
        print("No path provided!")
