import os
from ultralytics import YOLO

DATA = "/Users/aaryabalumalghe/yolo_multitask/datasets/detection/vehicles/data.yaml"

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA,
    epochs=20,
    imgsz=640,
    batch=16,
    name="detection_run",
    project="runs/detect",
    device="mps"
)

print("\nDetection training complete!")
print("Best model: runs/detect/detection_run/weights/best.pt")