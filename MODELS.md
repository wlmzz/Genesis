# Genesis AI - Model Files

**⚠️ Model files are NOT included in this repository due to their size (~176 MB total).**

## Quick Setup

Run the download script to get all required models:

```bash
./download_models.sh
```

Or follow the manual download instructions below.

---

## Required Models

### MediaPipe Models (~40 MB)

| Model | Size | Purpose | Download |
|-------|------|---------|----------|
| `face_landmarker.task` | 3.6 MB | 478 facial landmarks | [Link](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |
| `hand_landmarker.task` | 7.5 MB | 21 hand landmarks per hand | [Link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |
| `pose_landmarker_heavy.task` | 29 MB | 33 body pose landmarks | [Link](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task) |

### YOLO Models (~35 MB)

| Model | Size | Purpose | Download |
|-------|------|---------|----------|
| `yolov8n.pt` | 6.2 MB | Object detection (80 COCO classes) | [Link](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) |
| `yolov8n-pose.pt` | 6.5 MB | Pose estimation (nano) | [Link](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt) |
| `yolov8s-pose.pt` | 22 MB | Pose estimation (small) | [Link](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt) |

### dlib Models (~95 MB)

| Model | Size | Purpose | Download |
|-------|------|---------|----------|
| `shape_predictor_68_face_landmarks.dat` | 95 MB | 68 facial landmarks (legacy) | [Compressed](https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat.bz2) |

---

## Manual Download

If you prefer to download manually:

```bash
# MediaPipe Face
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# MediaPipe Hand
curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# MediaPipe Pose
curl -L -o pose_landmarker_heavy.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# YOLO
curl -L -o yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
curl -L -o yolov8n-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
curl -L -o yolov8s-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt"

# dlib (compressed)
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

---

## OWL-ViT Model

The OWL-ViT model (`google/owlvit-base-patch32`) is downloaded automatically via Hugging Face Transformers on first run:

- **Model**: `google/owlvit-base-patch32`
- **Size**: ~600 MB
- **Location**: `~/.cache/huggingface/hub/`
- **Auto-download**: Yes (on first use)

No manual download required!

---

## Llama Vision Model

The Llama 3.2 Vision model is managed via Ollama:

```bash
# Install if needed
ollama pull llama3.2-vision:11b
```

- **Model**: `llama3.2-vision:11b`
- **Size**: ~7.8 GB
- **Location**: Managed by Ollama
- **Purpose**: AI-powered lip reading

---

## Storage Requirements

- **Required models**: ~176 MB
- **OWL-ViT (first run)**: ~600 MB
- **Llama Vision (optional)**: ~7.8 GB
- **Total (full setup)**: ~8.5 GB

---

## Git LFS Alternative

If you prefer to use Git LFS for model versioning:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git lfs track "*.task"
git lfs track "*.dat"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for models"
```

Then commit model files normally.

---

## License Notes

- **MediaPipe models**: Apache 2.0
- **YOLO models**: AGPL-3.0
- **dlib models**: CC0 (Public Domain)
- **OWL-ViT**: Apache 2.0
- **Llama**: Meta Llama License

Please review individual model licenses before commercial use.
