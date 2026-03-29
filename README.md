# Parking Slot Detection (Raspberry Pi Ready)

This project counts free parking slots from a predefined slot layout.

Formula used:

`free_slots = total_slots - occupied_slots`

The script supports:

- Single image mode
- Live camera mode (works on Raspberry Pi with USB camera or Pi camera via V4L2)
- Headless mode (no GUI required)

## Files

- `parking_system.py` : main script
- `annotated_parking_coords.txt` : parking slot rectangles (`x1,y1,x2,y2[,flag]`)
- `yolo11n.pt` : YOLO model

## 1) Raspberry Pi Setup

On Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you get `Illegal instruction` on Raspberry Pi, it is usually from `torch/ultralytics` binary compatibility.
Use the OpenCV ONNX backend (no torch runtime required on Pi).

## 1.1) Illegal Instruction Fix (Recommended on Pi)

1. Export ONNX once on your laptop/PC (not on Pi):

```bash
yolo export model=yolo11n.pt format=onnx imgsz=416
```

1. Copy `yolo11n.onnx` to Raspberry Pi project folder.

1. Run script on Pi with OpenCV backend:

```bash
python parking_system.py images/test_obj.jpeg --backend opencv --model yolo11n.onnx --slots annotated_parking_coords.txt -o images/out.jpg --imgsz 416
```

For live camera on Pi with OpenCV backend:

```bash
python parking_system.py --camera --backend opencv --model yolo11n.onnx --slots annotated_parking_coords.txt -o images/latest_frame.jpg --camera-index 0 --imgsz 416 --save-interval 30
```

If your camera is PiCam, make sure V4L2 is available (`/dev/video0`).

## 2) Single Image Run

```bash
python parking_system.py images/test_obj.jpeg --model yolo11n.pt --slots annotated_parking_coords.txt -o images/out.jpg --device cpu --imgsz 416
```

If your slot file is wrong/misaligned, generate slots as a fixed 2x3 grid (6 total):

```bash
python parking_system.py images/test2.jpeg --model yolo11n.pt --grid-rows 3 --grid-cols 2 -o images/out.jpg --device cpu --imgsz 416
```

Optional: restrict grid to a region of interest:

```bash
python parking_system.py images/test2.jpeg --model yolo11n.pt --grid-rows 3 --grid-cols 2 --grid-roi 0,0,830,1280 -o images/out.jpg --device cpu --imgsz 416
```

If `ultralytics` works on your Pi, this command is fine.
If not, use `--backend opencv --model yolo11n.onnx`.

Expected output format:

`Total slots: 6; Occupied: 1; Free: 5`

## 3) Live Camera Run (Raspberry Pi)

```bash
python parking_system.py --camera --camera-index 0 --model yolo11n.pt --slots annotated_parking_coords.txt -o images/latest_frame.jpg --device cpu --imgsz 416 --save-interval 30
```

This saves one annotated frame every 30 frames and prints current counts.

If you have a display connected and want to see live window:

```bash
python parking_system.py --camera --show --camera-index 0 --model yolo11n.pt --slots annotated_parking_coords.txt --device cpu --imgsz 416
```

Press `q` to quit live display mode.

## 4) Performance Tips for Pi

- Keep `--imgsz` at `320` or `416` for better speed.
- Keep model as `yolo11n.pt` (nano model is faster).
- Use `--conf 0.3` or `0.4` to reduce weak detections.
- For parking-only logic with non-car objects, keep default classes (all classes).
- To only count cars, use: `--classes 2`.

## 5) User-Friendly Web App (Image Upload + IP Webcam)

The frontend is now connected to a backend API, so users do not need to build or paste terminal commands.

### 5.1 Start the web app

From project root:

```bash
pip install -r requirements.txt
python backend/app.py
```

Then open:

`http://localhost:5000`

### 5.2 What it supports

- Upload image file and process directly from browser
- Phone IP webcam URL capture (single frame)
- Live pull mode for IP webcam (process every N seconds)
- Adjustable model/backend/conf/imgsz/device and optional grid settings

### 5.3 Phone IP Webcam notes

- Use an app like "IP Webcam" on Android
- Keep phone and laptop on the same Wi-Fi network
- Typical URL examples:
  - `http://<phone-ip>:8080/video`
  - `http://<phone-ip>:8080/shot.jpg`
