# Steel Defect Detection System

## 🧩 Overview
AI-powered automated detection and classification of surface defects on steel sheets using computer vision. The system processes high-resolution images from top and bottom-mounted cameras to detect, classify, and log defects in real-time.

## 🎯 Goals & Objectives
- ✅ Detect and localize surface defects in steel sheets
- ✅ Classify defects into 6 categories (scratch, dent, bend, color defect, patch, hole)
- ✅ Process images from both top and bottom cameras
- ✅ Save defect metadata in structured format (CSV/DB)
- ✅ Support real-time and batch processing

## 📷 Input Sources
- Steel sheets on conveyor belt
- Top camera and bottom camera with synchronized capture
- Structured naming system: `01_up.jpg`, `01_down.jpg`, etc.

## 🧠 AI Model
- **Model**: YOLOv8 object detection (fine-tuned)
- **Input**: Camera images (720x720 resolution)
- **Output**: Bounding boxes + defect class + confidence

### Defect Categories
| ID | Defect Type    |
|----|----------------|
| 0  | Scratch        |
| 1  | Dent           |
| 2  | Bend           |
| 3  | Color Defect   |
| 4  | Hole           |
| 5  | Patch/Inclusion|

## 🗃️ Data Storage
Detection results saved in CSV format with:
- Image ID
- Camera (top/bottom)
- Defect type
- Location (x, y, width, height)
- Confidence score

## 💻 System Workflow
1. Conveyor moves steel sheet into camera zone
2. Cameras capture top and bottom images
3. Images fed to AI model
4. Model detects and classifies defects
5. Results saved in CSV or displayed on UI
6. Data logged for analysis

## 🧪 Tech Stack
- **Programming**: Python
- **CV Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV
- **Data Storage**: CSV / SQLite
- **Optional UI**: Streamlit

## 🚦 Project Structure
```
steel-defect-detection/
├── data/
│   ├── raw/              # Raw images
│   ├── labeled/          # Labeled dataset
│   └── processed/        # Processed images
├── models/
│   ├── training/         # Training scripts
│   └── weights/          # Model weights
├── src/
│   ├── detection/        # Detection pipeline
│   ├── preprocessing/    # Image preprocessing
│   └── utils/           # Utility functions
├── results/             # Detection results
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks
└── ui/                  # Optional UI components
```

## 🧪 Success Criteria
- Model accuracy ≥85%
- Detection time < 1 second per image
- Process ≥50 images/hour in batch mode
- Structured data output

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🚀 Usage
```bash
# Train model
python src/training/train.py

# Run detection
python src/detection/detect.py --input data/raw --output results/

# Launch UI (optional)
streamlit run ui/app.py
```

## 📊 Milestones
- [x] Phase 1: Project Setup
- [ ] Phase 2: Data Collection & Labeling
- [ ] Phase 3: Model Training
- [ ] Phase 4: Inference Pipeline
- [ ] Phase 5: Deployment & UI
