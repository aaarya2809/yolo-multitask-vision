import os
from ultralytics import YOLO

DATA = "/Users/aaryabalumalghe/yolo_multitask/datasets/pose/human_pose/data.yaml"

model = YOLO("yolov8n-pose.pt")

results = model.train(
    data=DATA,
    epochs=20,
    imgsz=640,
    batch=8,
    name="pose_run",
    project="runs/pose",
    device="mps"
)

print("\nPose training complete!")
print("Best model: runs/pose/pose_run/weights/best.pt")