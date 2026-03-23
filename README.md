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

If your camera is PiCam, make sure V4L2 is available (`/dev/video0`).

## 2) Single Image Run

```bash
python parking_system.py model_2/test_obj.jpeg --model yolo11n.pt --slots annotated_parking_coords.txt -o out.jpg --device cpu --imgsz 416
```

Expected output format:

`Total slots: 6; Occupied: 1; Free: 5`

## 3) Live Camera Run (Raspberry Pi)

```bash
python parking_system.py --camera --camera-index 0 --model yolo11n.pt --slots annotated_parking_coords.txt -o latest_frame.jpg --device cpu --imgsz 416 --save-interval 30
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
