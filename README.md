# Smart Parking (YOLOv8 + OpenCV)

Minimal Python project scaffold for a Smart Parking System using YOLOv8 and OpenCV.

Structure:

- `smart_parking/slots.py`: load/save slot polygons and point-in-polygon helper
- `smart_parking/logic.py`: `ParkingManager` to classify slots as Empty/Occupied
- `smart_parking/detector.py`: helper to parse YOLO results
- `main.py`: runner that loads a video, runs YOLOv8, and annotates frames
- `data/slots_example.json`: example slots definition
- `requirements.txt`: dependencies

Quick start:

1. Create a virtual env and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

2. Prepare or edit `data/slots_example.json` with polygons for your parking slots.

3. Run the app (webcam):

```bash
python main.py --video 0 --slots data/slots_example.json
```

Notes:
- The project uses a simple bbox-center-in-polygon test for occupancy. You can improve by checking box corners or IoU.
- Adjust `--model` and `--conf` flags in `main.py` as needed.

Training and evaluation
-----------------------

1. Prepare dataset in YOLO format under `dataset/` with images and `labels/` as described above.
2. Edit `data/vehicles.yaml` if your dataset path or classes differ.
3. Train:

```bash
python train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
```

4. Validate a trained model (replace path to your `best.pt`):

```bash
python val.py --weights runs/detect/parking_train/weights/best.pt
```

5. Run inference with saved weights:

```bash
python infer.py --weights runs/detect/parking_train/weights/best.pt --source sample_video.mp4 --save
```

6. Export model to ONNX:

```bash
python export_model.py --weights runs/detect/parking_train/weights/best.pt --format onnx
```

