import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("/Users/aaryabalumalghe/yolo_multitask/models/detect_best.pt")

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>YOLO Multi-Task Vision System</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 750px; margin: 50px auto; padding: 0 20px; background: #f9f9f9; }
    h1 { color: #222; font-size: 24px; }
    input[type=file] { margin: 12px 0; padding: 8px; }
    button { background: #0055cc; color: white; padding: 10px 28px; border: none; border-radius: 6px; cursor: pointer; font-size: 15px; }
    button:hover { background: #003fa3; }
    pre { background: #eef; padding: 14px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
    img { max-width: 100%; margin-top: 18px; border-radius: 8px; border: 1px solid #ddd; }
  </style>
</head>
<body>
  <h1>YOLO Multi-Task Vision System</h1>
  <p>Upload an image to run Vehicle Detection using YOLOv8n.</p>
  <form method="POST" action="/predict" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required><br>
    <button type="submit">Run Detection</button>
  </form>
  {% if result %}
    <h3>Detection Results:</h3>
    <pre>{{ result }}</pre>
    <img src="{{ img_path }}">
  {% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    results = model(fpath)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })
        r.save(filename=fpath)
    return render_template_string(
        HTML,
        result=str(detections) if detections else "No detections found",
        img_path="/" + fpath
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files["image"]
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    results = model(fpath)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 3)
            })
    return jsonify({"detections": detections, "count": len(detections)})

if __name__ == "__main__":
    print("Starting YOLO deployment at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
