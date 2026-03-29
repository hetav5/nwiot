import base64
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from parking_system import (  # noqa: E402
    generate_grid_slots,
    load_slots,
    process_frame,
    scale_slots_to_image,
)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

_DETECTOR_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_roi(raw: str) -> Optional[Tuple[int, int, int, int]]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in parts]
        return (x1, y1, x2, y2)
    except ValueError:
        return None


def _resolve_model_path(model_path: str) -> Path:
    candidate = (ROOT_DIR / model_path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Model file not found: {model_path}")


def _resolve_slots_path(slots_path: str) -> Path:
    candidate = (ROOT_DIR / slots_path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Slots file not found: {slots_path}")


def _load_detector(model_path: Path, backend: str, device: str):
    key = (backend, str(model_path), device)
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]

    if backend == "ultralytics":
        from ultralytics import YOLO

        detector = YOLO(str(model_path))
    else:
        detector = cv2.dnn.readNetFromONNX(str(model_path))

    _DETECTOR_CACHE[key] = detector
    return detector


def _get_scaled_slots(
    frame: np.ndarray,
    slots_path: str,
    grid_rows: int,
    grid_cols: int,
    grid_roi: Optional[Tuple[int, int, int, int]],
):
    h, w = frame.shape[:2]
    if grid_rows > 0 and grid_cols > 0:
        return generate_grid_slots(w, h, grid_rows, grid_cols, grid_roi)

    slots = load_slots(str(_resolve_slots_path(slots_path)))
    return scale_slots_to_image(slots, w, h)


def _encode_jpeg_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not ok:
        raise RuntimeError("Failed to encode output frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _extract_options(source: Dict[str, Any]) -> Dict[str, Any]:
    backend = source.get("backend", "ultralytics").strip() or "ultralytics"
    if backend not in {"ultralytics", "opencv"}:
        backend = "ultralytics"

    model = source.get("modelPath", "yolo11n.pt").strip() or "yolo11n.pt"
    slots = source.get("slotsPath", "annotated_parking_coords.txt").strip() or "annotated_parking_coords.txt"

    options = {
        "backend": backend,
        "model": model,
        "slots": slots,
        "conf": _to_float(source.get("conf", 0.25), 0.25),
        "imgsz": _to_int(source.get("imgsz", 416), 416),
        "device": source.get("device", "cpu").strip() or "cpu",
        "grid_rows": _to_int(source.get("gridRows", 0), 0),
        "grid_cols": _to_int(source.get("gridCols", 0), 0),
        "grid_roi": _parse_roi(source.get("gridRoi", "").strip()),
    }
    return options


def _run_parking_detection(frame: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    model_path = _resolve_model_path(options["model"])
    detector = _load_detector(model_path, options["backend"], options["device"])

    scaled_slots = _get_scaled_slots(
        frame,
        options["slots"],
        options["grid_rows"],
        options["grid_cols"],
        options["grid_roi"],
    )

    annotated, occupied, free, total = process_frame(
        frame,
        scaled_slots,
        detector,
        options["backend"],
        classes=None,
        conf=options["conf"],
        imgsz=options["imgsz"],
        device=options["device"],
    )

    return {
        "total": total,
        "occupied": occupied,
        "free": free,
        "image": _encode_jpeg_data_url(annotated),
    }


def _read_ipcam_frame(ip_url: str) -> np.ndarray:
    cap = cv2.VideoCapture(ip_url)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.size > 0:
            return frame

    with urllib.request.urlopen(ip_url, timeout=8) as response:
        payload = response.read()

    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode frame from IP webcam URL")
    return frame


@app.get("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "parking_dashboard.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/process-image")
def process_image():
    try:
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "No image file uploaded"}), 400

        upload = request.files["image"]
        raw = upload.read()
        if not raw:
            return jsonify({"ok": False, "error": "Uploaded image is empty"}), 400

        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"ok": False, "error": "Unsupported image format"}), 400

        options = _extract_options(request.form.to_dict())
        result = _run_parking_detection(frame, options)
        return jsonify({"ok": True, **result})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/process-ipcam")
def process_ipcam():
    try:
        payload = request.get_json(silent=True) or {}
        ip_url = (payload.get("ipUrl") or "").strip()
        if not ip_url:
            return jsonify({"ok": False, "error": "IP webcam URL is required"}), 400

        frame = _read_ipcam_frame(ip_url)
        options = _extract_options(payload)
        result = _run_parking_detection(frame, options)
        return jsonify({"ok": True, **result})
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
