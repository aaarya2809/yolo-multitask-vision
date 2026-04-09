import os
from ultralytics import YOLO

DATA = "/Users/aaryabalumalghe/yolo_multitask/datasets/obb/text_detection_obb/data.yaml"

model = YOLO("yolov8n-obb.pt")

results = model.train(
    data=DATA,
    epochs=20,
    imgsz=640,
    batch=8,
    name="obb_run",
    project="runs/obb",
    device="mps"
)

print("\nOBB training complete!")
print("Best model: runs/obb/obb_run/weights/best.pt")