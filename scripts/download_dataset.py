"""
Dataset Download and Setup Script for Steel Defect Detection
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
import json


def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            pbar.update(size)


def setup_neu_dataset():
    """Download and setup NEU Steel Surface Defect Dataset."""
    print("🏭 Setting up NEU Steel Surface Defect Dataset...")
    
    # Create directories
    data_dir = Path("data/labeled")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # NEU dataset info
    neu_classes = {
        'crazing': 0,      # -> scratch
        'inclusion': 1,    # -> patch  
        'patches': 2,      # -> patch
        'pitted_surface': 3, # -> hole
        'rolled-in_scale': 4, # -> color_defect
        'scratches': 5     # -> scratch
    }
    
    # Map to our defect classes
    our_classes = {
        0: 'scratch',
        1: 'dent',
        2: 'bend', 
        3: 'color_defect',
        4: 'hole',
        5: 'patch'
    }
    
    class_mapping = {
        'crazing': 0,        # scratch
        'inclusion': 5,      # patch
        'patches': 5,        # patch
        'pitted_surface': 4, # hole
        'rolled-in_scale': 3, # color_defect
        'scratches': 0       # scratch
    }
    
    print("📋 Dataset will be mapped to our classes:")
    for neu_class, our_id in class_mapping.items():
        print(f"   {neu_class} -> {our_classes[our_id]}")
    
    # Note: Since we can't directly download NEU dataset (requires academic access),
    # we'll create a setup structure and instructions
    
    setup_info = {
        "dataset_name": "NEU Steel Surface Defect Dataset",
        "classes": our_classes,
        "class_mapping": class_mapping,
        "instructions": [
            "1. Download NEU dataset from: https://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html",
            "2. Extract the dataset",
            "3. Run: python scripts/convert_neu_dataset.py --input <neu_path> --output data/labeled",
            "4. The dataset will be converted to YOLO format automatically"
        ],
        "alternative_sources": [
            "Kaggle: Search for 'NEU Steel Surface Defect'",
            "GitHub: Several repositories host this dataset",
            "Academic papers often provide links"
        ]
    }
    
    # Save setup info
    with open("data/dataset_info.json", "w") as f:
        json.dump(setup_info, f, indent=2)
    
    print("\n📁 Dataset structure created!")
    print("📄 Instructions saved to: data/dataset_info.json")
    
    return setup_info


def create_sample_dataset():
    """Create a small sample dataset for testing."""
    print("🧪 Creating sample dataset for testing...")
    
    # Create sample directory structure
    sample_dir = Path("data/sample")
    images_dir = sample_dir / "images"
    labels_dir = sample_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample YOLO dataset.yaml
    dataset_yaml = {
        'path': str(sample_dir.absolute()),
        'train': 'images',
        'val': 'images',
        'names': {
            0: 'scratch',
            1: 'dent',
            2: 'bend',
            3: 'color_defect', 
            4: 'hole',
            5: 'patch'
        }
    }
    
    with open(sample_dir / "dataset.yaml", "w") as f:
        import yaml
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"📁 Sample dataset structure created at: {sample_dir}")
    print("🔧 Place your images and labels in the respective folders")
    print("📄 dataset.yaml file created for YOLO training")
    
    return sample_dir


def download_severstal_dataset():
    """Instructions for downloading Severstal dataset."""
    print("🏭 Severstal Steel Defect Detection Dataset Setup")
    print("=" * 50)
    print("📋 This dataset requires Kaggle API access")
    print("\n🚀 Setup Instructions:")
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Setup Kaggle credentials: https://www.kaggle.com/docs/api")
    print("3. Run: kaggle competitions download -c severstal-steel-defect-detection")
    print("4. Extract and convert to YOLO format")
    
    return "Severstal dataset requires manual setup"


if __name__ == "__main__":
    print("🔧 Steel Defect Dataset Setup")
    print("=" * 40)
    
    choice = input("""
Choose dataset option:
1. NEU Steel Surface Defect Dataset (Recommended)
2. Create sample dataset structure
3. Severstal dataset instructions
4. All of the above

Enter choice (1-4): """)
    
    if choice == "1":
        setup_neu_dataset()
    elif choice == "2":
        create_sample_dataset()
    elif choice == "3":
        download_severstal_dataset()
    elif choice == "4":
        setup_neu_dataset()
        create_sample_dataset()
        download_severstal_dataset()
    else:
        print("Invalid choice!")
    
    print("\n✅ Dataset setup completed!")
    print("📖 Check data/dataset_info.json for detailed instructions")
