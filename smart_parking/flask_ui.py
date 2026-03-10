import os
import sys
import uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

# Ensure project root is on sys.path so `import smart_parking.*` works
APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from smart_parking.run_parking_check import check_image, visualize_and_save
from smart_parking.stream_processor import CameraStream

APP_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = APP_ROOT / "runs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(16)

# global camera stream holder
_CAMERA_STREAM = None


def list_sample_images(limit=20):
    base = APP_ROOT / "dataset"
    imgs = []
    for split in ("train", "valid", "test"):
        p = base / split
        if p.exists():
            # return paths relative to dataset folder so URLs can be served
            imgs.extend([os.path.relpath(str(x), start=str(base)) for x in p.glob("*.jpg")])
    return imgs[:limit]


@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    data_dir = APP_ROOT / 'dataset'
    return send_from_directory(str(data_dir), filename)


@app.route("/runs/<path:filename>")
def serve_runs(filename):
    # Serve files under the runs folder
    runs_dir = APP_ROOT / "runs"
    return send_from_directory(str(runs_dir), filename)


@app.route("/", methods=["GET"])
def index():
    samples = list_sample_images(50)
    return render_template("index.html", samples=samples)


@app.route("/detect", methods=["POST"])
def detect():
    model = request.form.get("model", "yolov8m.pt")
    conf = float(request.form.get("conf", 0.25))
    pxpm = float(request.form.get("pxpm", 100.0))
    slots = request.form.get("slots", "data/slots_example.json")

    # determine image source: uploaded file or selected sample
    uploaded = request.files.get("file")
    selected = request.form.get("selected")

    if uploaded and uploaded.filename:
        uid = uuid.uuid4().hex
        filename = f"upload_{uid}.jpg"
        outpath = UPLOAD_DIR / filename
        uploaded.save(outpath)
        image_path = str(outpath)
    elif selected:
        image_path = str(APP_ROOT / "dataset" / selected)
    else:
        flash("No image selected or uploaded")
        return redirect(url_for("index"))

    # resolve and validate slots and model paths
    slots_path = slots if os.path.isabs(slots) else str(APP_ROOT / slots)
    if not os.path.exists(slots_path):
        flash(f"Slots file not found: {slots_path}")
        return redirect(url_for("index"))

    model_path = model if os.path.isabs(model) else str(APP_ROOT / model)
    if not os.path.exists(model_path):
        flash(f"Model weights not found: {model_path}")
        return redirect(url_for("index"))

    # run check and save outputs
    out_dir = APP_ROOT / "runs" / f"flask_{uuid.uuid4().hex}"
    out_dir = str(out_dir)
    report, dets = check_image(image_path, model_path, slots_path, conf_thresh=conf, bev_px_per_meter=pxpm)
    annotated, report_json = visualize_and_save(image_path, slots_path, report, dets, out_dir)

    # compute relative paths for template
    annotated_rel = os.path.relpath(annotated, start=str(APP_ROOT)).replace('\\', '/')
    report_rel = os.path.relpath(report_json, start=str(APP_ROOT)).replace('\\', '/')

    return render_template("result.html", annotated=annotated_rel, report_path=report_rel, report=report)


@app.route('/start_stream', methods=['POST'])
def start_stream():
    global _CAMERA_STREAM
    source = request.form.get('camera_url')
    model = request.form.get('model', 'yolov8m.pt')
    slots = request.form.get('slots', 'data/slots_example.json')
    conf = float(request.form.get('conf', 0.25))
    pxpm = float(request.form.get('pxpm', 100.0))
    skip = int(request.form.get('frame_skip', 5))

    if not source:
        flash('Provide camera URL or device index')
        return redirect(url_for('index'))

    if _CAMERA_STREAM and getattr(_CAMERA_STREAM, 'source', None) == source:
        flash('Stream already running')
        return redirect(url_for('index'))

    # stop existing
    if _CAMERA_STREAM:
        try:
            _CAMERA_STREAM.stop()
        except Exception:
            pass

    # resolve model and slots paths
    slots_path = slots if os.path.isabs(slots) else str(APP_ROOT / slots)
    model_path = model if os.path.isabs(model) else str(APP_ROOT / model)
    if not os.path.exists(model_path):
        flash(f"Model weights not found: {model_path}")
        return redirect(url_for('index'))
    if not os.path.exists(slots_path):
        flash(f"Slots file not found: {slots_path}")
        return redirect(url_for('index'))

    # convert numeric camera index
    try:
        if isinstance(source, str) and source.isdigit():
            source_val = int(source)
        else:
            source_val = source
    except Exception:
        source_val = source

    _CAMERA_STREAM = CameraStream(source=source_val, model_path=model_path, slots_json=slots_path, conf=conf, pxpm=pxpm, frame_skip=skip)
    _CAMERA_STREAM.start()
    flash('Stream started')
    return redirect(url_for('index'))


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global _CAMERA_STREAM
    if _CAMERA_STREAM:
        _CAMERA_STREAM.stop()
        _CAMERA_STREAM = None
        flash('Stream stopped')
    else:
        flash('No active stream')
    return redirect(url_for('index'))


@app.route('/latest_report')
def latest_report():
    out_dir = APP_ROOT / 'runs' / 'parking_stream'
    path = out_dir / 'report.json'
    if path.exists():
        return send_from_directory(str(out_dir), 'report.json')
    return ({}, 204)


@app.route('/latest_image')
def latest_image():
    out_dir = APP_ROOT / 'runs' / 'parking_stream'
    path = out_dir / 'annotated.jpg'
    if path.exists():
        return send_from_directory(str(out_dir), 'annotated.jpg')
    return ('', 204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=True)
