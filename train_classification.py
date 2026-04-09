import os
from ultralytics import YOLO

DATA = "/Users/aaryabalumalghe/yolo_multitask/datasets/classification/plant_disease"

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=DATA,
    epochs=20,
    imgsz=224,
    batch=32,
    name="classification_run",
    project="runs/classify",
    device="mps"
)

print("\nClassification training complete!")
print("Best model: runs/classify/classification_run/weights/best.pt")