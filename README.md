# YOLO Multi-Task Vision System

Deep Learning Lab Assignment — Multi-Task Computer Vision using YOLOv8

## System
- Device: MacBook Air — Apple M4
- Python: 3.11.13
- PyTorch: 2.11.0 (MPS)
- Ultralytics: 8.4.36

## Tasks
| Part | Task | Model | Result |
|------|------|-------|--------|
| A | Vehicle Detection | yolov8n | mAP50: 0.009 |
| B | Plant Disease Classification | yolov8n-cls | Top-1: 100% |
| C | Human Pose Estimation | yolov8n-pose | mAP50: 0.955 |
| D | Text Detection OBB | yolov8n-obb | mAP50: 0.819 |

## Datasets (Roboflow)
- Detection: vehicles.v1i.yolov8
- Classification: Plant Disease.v1i.folder
- Pose: Human Pose.v1i.yolov8
- OBB: Text Detection.v1i.yolov8-obb

## Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install ultralytics flask streamlit opencv-python pillow
```

## Training
```bash
python train_detection.py
python train_classification.py
python train_pose.py
python train_obb.py
```

## Run Streamlit App
```bash
https://yolo-multitask-vision.streamlit.app/
```

## Run Flask App
```bash
cd deployment && python app.py
```

## Project Structure
yolo_multitask/
├── train_detection.py
├── train_classification.py
├── train_pose.py
├── train_obb.py
├── run_all_tasks.py
├── app_streamlit.py
├── deployment/app.py
├── models/          (not pushed — too large)
├── datasets/        (not pushed — from Roboflow)
└── runs/            (not pushed — training outputs)
