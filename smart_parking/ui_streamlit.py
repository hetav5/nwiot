import streamlit as st
from pathlib import Path
import random
import io
import json

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from smart_parking.detector import parse_results
from smart_parking.slots import load_slots, detection_inside_slot
from smart_parking.transform import get_birds_eye, bbox_size_in_bev


st.set_page_config(page_title="Smart Parking Tester", layout="wide")

st.title("Smart Parking — Test UI")

workspace = Path(".")
dataset_paths = [workspace / "dataset" / p for p in ("train", "valid", "test")]

model_path = st.sidebar.text_input("YOLO weights path", value="yolov8m.pt")
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
pxpm = st.sidebar.number_input("BEV pixels-per-meter", value=100.0)
slots_json = st.sidebar.text_input(
    "Slots JSON", value="data/slots_example.json")

st.sidebar.markdown("---")
st.sidebar.write("Pick a random image or upload one:")

random_src = st.sidebar.selectbox(
    "Random source", ["dataset/train", "dataset/valid", "dataset/test"])
if st.sidebar.button("Pick random image"):
    folder = workspace / Path(random_src)
    imgs = list(folder.glob("*.jpg"))
    if imgs:
        chosen = random.choice(imgs)
        st.session_state["image_path"] = str(chosen)
    else:
        st.sidebar.error("No images found in chosen folder")

uploaded = st.sidebar.file_uploader(
    "Or upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    # read into session as bytes
    st.session_state["uploaded_bytes"] = uploaded.getvalue()
    st.session_state["image_path"] = None


def run_detection_on_bytes(img_bytes, model_path, conf_thresh):
    arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    model = YOLO(model_path)
    # ultralytics accepts numpy arrays as source
    results = model.predict(source=img, conf=conf_thresh, save=False)
    dets = parse_results(results)
    return img, dets


def run_detection_on_path(img_path, model_path, conf_thresh):
    model = YOLO(model_path)
    results = model.predict(source=str(img_path), conf=conf_thresh, save=False)
    dets = parse_results(results)
    img = cv2.imread(str(img_path))
    return img, dets


col1, col2 = st.columns([1, 1])

image_path = st.session_state.get("image_path", None)
if image_path:
    col1.markdown(f"**Selected image:** {image_path}")
elif st.session_state.get("uploaded_bytes") is not None:
    col1.markdown("**Selected image:** uploaded file")
else:
    col1.info("No image selected — pick random or upload one")

if st.button("Run detection"):
    with st.spinner("Running model..."):
        try:
            if st.session_state.get("uploaded_bytes") is not None:
                img, dets = run_detection_on_bytes(
                    st.session_state["uploaded_bytes"], model_path, conf)
            elif image_path:
                img, dets = run_detection_on_path(image_path, model_path, conf)
            else:
                st.error("No image selected")
                st.stop()

            # visualize
            vis = img.copy()
            for d in dets:
                x1, y1, x2, y2 = map(int, d["xyxy"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, f"c:{d['class']} {d['conf']:.2f}", (x1,
                            y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # slots report if slots file exists
            report = None
            slots_file = Path(slots_json)
            if slots_file.exists():
                slots = load_slots(str(slots_file))
                report = []
                for slot in slots:
                    occ = False
                    for d in dets:
                        if detection_inside_slot(d["xyxy"], slot.get("polygon", [])):
                            occ = True
                            break
                    report.append({"slot": slot.get("name"), "occupied": occ})

            # convert BGR->RGB for display
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(vis_rgb)

            col1.image(pil_img, caption="Annotated", use_column_width=True)
            if report is not None:
                col2.subheader("Slot occupancy report")
                col2.json(report)
            col2.subheader("Detections")
            col2.write(dets)

        except Exception as e:
            st.error(f"Error running detection: {e}")

st.markdown("---")
st.markdown("Preview dataset folders:")
for p in dataset_paths:
    st.write(f"{p}: {len(list(p.glob('*.jpg')))} images")
