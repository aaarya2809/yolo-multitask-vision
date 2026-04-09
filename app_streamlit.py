import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(
    page_title="YOLO Multi-Task Vision System",
    page_icon="🤖",
    layout="wide"
)

LOCAL_BASE = "/Users/aaryabalumalghe/yolo_multitask/models"
IS_LOCAL = os.path.exists(LOCAL_BASE)

@st.cache_resource
def load_models():
    if IS_LOCAL:
        return {
            "detection":      YOLO(f"{LOCAL_BASE}/detect_best.pt"),
            "classification": YOLO(f"{LOCAL_BASE}/classify_best.pt"),
            "pose":           YOLO(f"{LOCAL_BASE}/pose_best.pt"),
            "obb":            YOLO(f"{LOCAL_BASE}/obb_best.pt"),
        }
    else:
        return {
            "detection":      YOLO("yolov8n.pt"),
            "classification": YOLO("yolov8n-cls.pt"),
            "pose":           YOLO("yolov8n-pose.pt"),
            "obb":            YOLO("yolov8n-obb.pt"),
        }

models = load_models()

st.markdown("""
<style>
    .main-header {
        background: #1a3c6e;
        padding: 24px 32px; border-radius: 14px;
        margin-bottom: 24px; color: white;
    }
    .main-header h1 { font-size: 26px; font-weight: 700; margin: 0 0 6px; }
    .main-header p  { font-size: 14px; opacity: 0.8; margin: 0; }
    .badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        color: white; padding: 3px 10px; border-radius: 20px;
        font-size: 12px; margin: 4px 4px 0 0;
    }
    .metric-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
    .metric-card {
        background: white; border: 0.5px solid #e2e8f0;
        border-radius: 12px; padding: 16px 20px;
        flex: 1; min-width: 120px; text-align: center;
    }
    .metric-val { font-size: 22px; font-weight: 700; color: #1a3c6e; }
    .metric-lbl { font-size: 12px; color: #64748b; margin-top: 2px; }
    .task-desc {
        background: #eff6ff; border: 0.5px solid #bfdbfe;
        border-radius: 8px; padding: 10px 14px;
        font-size: 13px; color: #1e40af; margin-bottom: 16px;
    }
    .det-item {
        background: #f8fafc; border: 0.5px solid #e2e8f0;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .conf-pill {
        background: #dcfce7; color: #166534;
        padding: 2px 10px; border-radius: 20px; font-size: 12px;
    }
    .stButton > button {
        background: #1a3c6e !important; color: white !important;
        border: none !important; border-radius: 10px !important;
        padding: 10px 24px !important; font-weight: 500 !important;
        width: 100% !important;
    }
    .stButton > button:hover { background: #1e4d8c !important; }
</style>
""", unsafe_allow_html=True)

mode_label = "Local — Apple M4 (trained models)" if IS_LOCAL else "Streamlit Cloud — pretrained YOLOv8n"

st.markdown(f"""
<div class="main-header">
  <h1>YOLO Multi-Task Vision System</h1>
  <p>Deep Learning Lab Assignment — {mode_label}</p>
  <div style="margin-top:10px">
    <span class="badge">YOLOv8n</span>
    <span class="badge">4 Tasks</span>
    <span class="badge">Ultralytics</span>
    <span class="badge">{"Apple M4 MPS" if IS_LOCAL else "Streamlit Cloud"}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
  <div class="metric-card"><div class="metric-val">4</div><div class="metric-lbl">Models trained</div></div>
  <div class="metric-card"><div class="metric-val">100%</div><div class="metric-lbl">Cls accuracy</div></div>
  <div class="metric-card"><div class="metric-val">0.955</div><div class="metric-lbl">Pose mAP50</div></div>
  <div class="metric-card"><div class="metric-val">0.819</div><div class="metric-lbl">OBB mAP50</div></div>
</div>
""", unsafe_allow_html=True)

TASKS = {
    "Vehicle Detection":            ("detection",      "Upload a road or traffic image to detect vehicles"),
    "Plant Disease Classification": ("classification", "Upload a plant leaf image to identify disease"),
    "Human Pose Estimation":        ("pose",           "Upload an image with people to detect body keypoints"),
    "Text Detection (OBB)":         ("obb",            "Upload an image with text to detect rotated text regions"),
}

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### Select task")
    task_name = st.radio("task", list(TASKS.keys()), label_visibility="collapsed")
    task_key, task_desc = TASKS[task_name]

    st.markdown(f'<div class="task-desc">{task_desc}</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

    run = st.button("Run inference", disabled=not uploaded)

with col_right:
    st.markdown("### Result")

    if uploaded and run:
        image = Image.open(uploaded).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        model = models[task_key]

        with st.spinner(f"Running {task_name}..."):
            results = model(tmp_path)

        r = results[0]

        if task_key == "classification":
            st.success("Classification complete!")
            names        = r.names
            probs        = r.probs
            top5_indices = probs.top5
            top5_confs   = probs.top5conf.tolist()

            st.markdown("**Top predictions:**")
            for idx, conf in zip(top5_indices, top5_confs):
                class_name = names[idx]
                st.markdown(
                    f'<div class="det-item">'
                    f'<span style="font-weight:500;font-size:14px">{class_name}</span>'
                    f'<span class="conf-pill">{conf*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.image(image, caption="Input image", use_container_width=True)

        elif task_key == "pose":
            annotated     = r.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Pose estimation output", use_container_width=True)
            st.success("Pose estimation complete!")
            boxes = r.boxes
            st.markdown(f"**{len(boxes)} person(s) detected**")
            for i, box in enumerate(boxes):
                conf = float(box.conf)
                st.markdown(
                    f'<div class="det-item">'
                    f'<span style="font-weight:500">Person {i+1}</span>'
                    f'<span class="conf-pill">{conf*100:.1f}% conf</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            if r.keypoints is not None:
                st.info("17 keypoints detected per person (COCO format)")

        elif task_key == "detection":
            annotated     = r.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detection output", use_container_width=True)
            st.success("Vehicle detection complete!")
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                st.markdown(f"**{len(boxes)} object(s) detected**")
                for box in boxes:
                    cls_name = r.names[int(box.cls)]
                    conf     = float(box.conf)
                    st.markdown(
                        f'<div class="det-item">'
                        f'<span style="font-weight:500">{cls_name}</span>'
                        f'<span class="conf-pill">{conf*100:.1f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No objects detected — try a different image.")

        elif task_key == "obb":
            annotated     = r.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="OBB detection output", use_container_width=True)
            st.success("Text detection (OBB) complete!")
            obbs = r.obb
            if obbs is not None and len(obbs) > 0:
                st.markdown(f"**{len(obbs)} text region(s) detected**")
                for box in obbs:
                    cls_name = r.names[int(box.cls)]
                    conf     = float(box.conf)
                    st.markdown(
                        f'<div class="det-item">'
                        f'<span style="font-weight:500">{cls_name}</span>'
                        f'<span class="conf-pill">{conf*100:.1f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No text regions detected — try an image with clear text.")

        os.unlink(tmp_path)

    elif not uploaded:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    min-height:320px;flex-direction:column;gap:12px;color:#94a3b8;
                    border:1.5px dashed #e2e8f0;border-radius:12px;margin-top:8px">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <path d="M21 15l-5-5L5 21"/>
          </svg>
          <p style="font-size:14px">Select a task and upload an image</p>
          <p style="font-size:12px">Annotated results will appear here</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            font-size:12px;color:#94a3b8;padding:4px 0">
  <span>YOLO Multi-Task Vision System — Deep Learning Lab Assignment</span>
  <span>Aarya Balumalghe | Apple M4 | Ultralytics 8.4.36</span>
</div>
""", unsafe_allow_html=True)
