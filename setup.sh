#!/bin/bash
# Genesis - Setup Script for macOS

echo "=================================================="
echo "Genesis - Setup & Installation"
echo "=================================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Install Homebrew dependencies for dlib
echo ""
echo "Installing system dependencies (requires Homebrew)..."
if command -v brew &> /dev/null; then
    echo "  Installing cmake, boost..."
    brew install cmake boost boost-python3
else
    echo "⚠️  Homebrew not found. Please install manually:"
    echo "    cmake, boost, boost-python3"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing Python dependencies..."
echo "⚠️  This may take several minutes (downloading models)..."
pip install -r requirements.txt

# Download YOLO model
echo ""
echo "Downloading YOLOv8 model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/faces data/outputs

echo ""
echo "=================================================="
echo "✓ Setup completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Register faces: python app/face_register.py --person_id john_doe"
echo "  3. Define zones: python app/zone_editor.py"
echo "  4. Run analysis: python app/run_camera.py"
echo "  5. View dashboard: streamlit run app/dashboard.py"
echo ""
