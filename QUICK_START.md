# Genesis - Quick Start Guide

**WARNING: INTERNAL TESTING ONLY - NOT GDPR COMPLIANT**

## Quick Setup (5 minutes)

### 1. Installation

```bash
# Run automated setup (macOS)
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Register faces in the database

```bash
# Register a person
python app/face_register.py --person_id john_doe

# Register multiple people
python app/face_register.py --person_id alice
python app/face_register.py --person_id bob
```

### 3. Define tracking zones

```bash
# Open zone editor
python app/zone_editor.py --source 0

# Click to define polygons
# Press 't' to save zone
# Press 's' to save file
```

### 4. Start analysis

**From webcam:**
```bash
python app/run_camera.py --cam 0
```

**From video:**
```bash
python app/run_video.py --video path/to/video.mp4
```

### 5. View dashboard

```bash
# In a new terminal
source .venv/bin/activate
streamlit run app/dashboard.py
```

Open browser: http://localhost:8501

---

## Configuration

Edit `configs/settings.yaml` to customize:

- **Facial recognition**: enable/disable, change model, threshold
- **Detection**: confidence, IOU, YOLO model
- **Alerts**: queue thresholds, wait times
- **Export**: metrics save interval

---

## Output

- **CSV**: `data/outputs/metrics.csv` - aggregated metrics
- **JSONL**: `data/outputs/metrics.jsonl` - JSON metrics
- **Database**: `data/outputs/identities.db` - identity timeline
- **Faces**: `data/faces/` - embeddings database

---

## Troubleshooting

### dlib installation fails on macOS

```bash
brew install cmake boost boost-python3
pip install dlib
```

### YOLO model not found

```bash
# Download manually
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Webcam not working

- Grant camera permissions in System Settings > Privacy > Camera
- Try --cam 1 or --cam 2

### Slow performance

Reduce in `settings.yaml`:
- `resize_width`: 640 instead of 960
- `update_embeddings_every`: 60 instead of 30

---

## Available Face Recognition Models

In `settings.yaml` → `face_recognition.model`:

- **Facenet512** (default) - Fast and accurate
- **VGG-Face** - High accuracy
- **ArcFace** - Excellent performance
- **OpenFace** - Lightweight
- **DeepFace** - Facebook model

---

For complete documentation see **README.md**
