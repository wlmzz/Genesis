#!/bin/bash
# Genesis AI - Model Download Script
# Downloads all required model files

set -e

echo "==================================================================="
echo "Genesis AI - Downloading Required Models"
echo "==================================================================="

# Create models directory if it doesn't exist
mkdir -p models

echo ""
echo "📦 Downloading MediaPipe models..."

# Face Landmarker (3.6MB)
if [ ! -f "face_landmarker.task" ]; then
    echo "  → Face Landmarker..."
    curl -L -o face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    echo "  ✓ Face Landmarker downloaded"
else
    echo "  ✓ Face Landmarker already exists"
fi

# Hand Landmarker (7.5MB)
if [ ! -f "hand_landmarker.task" ]; then
    echo "  → Hand Landmarker..."
    curl -L -o hand_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    echo "  ✓ Hand Landmarker downloaded"
else
    echo "  ✓ Hand Landmarker already exists"
fi

# Pose Landmarker Heavy (29MB)
if [ ! -f "pose_landmarker_heavy.task" ]; then
    echo "  → Pose Landmarker (Heavy)..."
    curl -L -o pose_landmarker_heavy.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    echo "  ✓ Pose Landmarker downloaded"
else
    echo "  ✓ Pose Landmarker already exists"
fi

echo ""
echo "🎯 Downloading YOLO models..."

# YOLOv8n (6.2MB)
if [ ! -f "yolov8n.pt" ]; then
    echo "  → YOLOv8 Nano..."
    curl -L -o yolov8n.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    echo "  ✓ YOLOv8n downloaded"
else
    echo "  ✓ YOLOv8n already exists"
fi

# YOLOv8n-pose (6.5MB)
if [ ! -f "yolov8n-pose.pt" ]; then
    echo "  → YOLOv8 Nano Pose..."
    curl -L -o yolov8n-pose.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
    echo "  ✓ YOLOv8n-pose downloaded"
else
    echo "  ✓ YOLOv8n-pose already exists"
fi

# YOLOv8s-pose (22MB)
if [ ! -f "yolov8s-pose.pt" ]; then
    echo "  → YOLOv8 Small Pose..."
    curl -L -o yolov8s-pose.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt"
    echo "  ✓ YOLOv8s-pose downloaded"
else
    echo "  ✓ YOLOv8s-pose already exists"
fi

echo ""
echo "👤 Downloading dlib face landmarks..."

# dlib face predictor (95MB compressed, 99MB uncompressed)
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "  → dlib 68 Face Landmarks (may take a while)..."

    # Download compressed file
    if [ ! -f "shape_predictor_68_face_landmarks.dat.bz2" ]; then
        curl -L -o shape_predictor_68_face_landmarks.dat.bz2 \
            "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    fi

    # Decompress
    echo "  → Extracting..."
    bunzip2 -k shape_predictor_68_face_landmarks.dat.bz2
    echo "  ✓ dlib predictor extracted"
else
    echo "  ✓ dlib predictor already exists"
fi

echo ""
echo "==================================================================="
echo "✅ All models downloaded successfully!"
echo "==================================================================="
echo ""
echo "Total size: ~176 MB"
echo ""
echo "You can now run Genesis AI:"
echo "  python app/run_camera_ai.py"
echo ""
