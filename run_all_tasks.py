import os
from ultralytics import YOLO

os.chdir("/Users/aaryabalumalghe/yolo_multitask")

detect_model = YOLO("models/detect_best.pt")
cls_model    = YOLO("models/classify_best.pt")
pose_model   = YOLO("models/pose_best.pt")
obb_model    = YOLO("models/obb_best.pt")

def get_first_image(folder):
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, f)
    return None

print("\n=== Part A: Vehicle Detection ===")
img = get_first_image("datasets/detection/vehicles/test/images")
print(f"Testing on: {img}")
r = detect_model(img)
r[0].save("output_detection.jpg")
print("Saved: output_detection.jpg")

print("\n=== Part B: Plant Disease Classification ===")
img = get_first_image("datasets/classification/plant_disease/test")
print(f"Testing on: {img}")
r = cls_model(img)
print("Top 5 predictions:", r[0].probs.top5)
r[0].save("output_classification.jpg")
print("Saved: output_classification.jpg")

print("\n=== Part C: Human Pose Estimation ===")
img = get_first_image("datasets/pose/human_pose/test/images")
print(f"Testing on: {img}")
r = pose_model(img)
r[0].save("output_pose.jpg")
print("Saved: output_pose.jpg")

print("\n=== Part D: Text Detection OBB ===")
img = get_first_image("datasets/obb/text_detection_obb/test/images")
print(f"Testing on: {img}")
r = obb_model(img)
r[0].save("output_obb.jpg")
print("Saved: output_obb.jpg")

print("\nAll 4 tasks complete!")
print("Check: output_detection.jpg, output_classification.jpg, output_pose.jpg, output_obb.jpg")
