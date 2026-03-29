import cv2
import os
import argparse
import numpy as np


def load_slots(path):
    slots = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Slots file not found: {path}")
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(",") if p.strip()]
            if len(parts) < 4:
                continue
            try:
                x1 = int(parts[0])
                y1 = int(parts[1])
                x2 = int(parts[2])
                y2 = int(parts[3])
                slots.append((x1, y1, x2, y2))
            except ValueError:
                continue
    return slots


def scale_slots_to_image(slots, img_w, img_h):
    if not slots:
        return slots
    max_x = max(r[2] for r in slots)
    max_y = max(r[3] for r in slots)
    if max_x == 0 or max_y == 0:
        return slots
    scale_x = img_w / max_x
    scale_y = img_h / max_y
    scale = min(scale_x, scale_y)
    if abs(scale - 1.0) < 1e-6:
        return slots
    scaled = []
    for (x1, y1, x2, y2) in slots:
        sx1 = int(x1 * scale)
        sy1 = int(y1 * scale)
        sx2 = int(x2 * scale)
        sy2 = int(y2 * scale)
        scaled.append((sx1, sy1, sx2, sy2))
    return scaled


def generate_grid_slots(img_w, img_h, rows, cols, roi=None):
    if roi is None:
        x1, y1, x2, y2 = 0, 0, img_w, img_h
    else:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

    grid_w = x2 - x1
    grid_h = y2 - y1
    cell_w = grid_w / float(cols)
    cell_h = grid_h / float(rows)

    slots = []
    for r in range(rows):
        for c in range(cols):
            sx1 = int(x1 + c * cell_w)
            sy1 = int(y1 + r * cell_h)
            sx2 = int(x1 + (c + 1) * cell_w)
            sy2 = int(y1 + (r + 1) * cell_h)
            slots.append((sx1, sy1, sx2, sy2))
    return slots


def rects_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2:
        return False
    return True


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def rect_area(r):
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def visual_occupied_slot_candidates(frame, slots):
    """Find occupied-looking slots using adaptive texture outliers.

    Uses Laplacian variance per slot and computes a per-frame threshold from
    the slot score distribution, avoiding fixed scene-specific constants.
    """
    slot_scores = []
    for i, s in enumerate(slots):
        x1, y1, x2, y2 = s
        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        slot_scores.append((i, lap_var))

    if not slot_scores:
        return []

    values = np.array([v for _, v in slot_scores], dtype=np.float32)
    q75 = float(np.quantile(values, 0.75))
    med = float(np.median(values))
    adaptive_threshold = max(q75, med * 3.0)

    candidates = [(idx, score)
                  for idx, score in slot_scores if score >= adaptive_threshold]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates]


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def detect_with_ultralytics(frame, detector, classes, conf, imgsz, device):
    results = detector.predict(frame, conf=conf, imgsz=imgsz,
                               device=device, verbose=False)
    detected_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            if classes is not None and len(classes) > 0 and cls not in classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_clamped = clamp_box(
                x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            if box_clamped is not None:
                detected_boxes.append(box_clamped)
    return detected_boxes


def detect_with_opencv_onnx(frame, net, classes, conf, imgsz):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0 / 255.0, size=(imgsz, imgsz), swapRB=True, crop=False
    )
    net.setInput(blob)
    out = net.forward()

    if out.ndim == 3:
        out = out[0]
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T

    boxes_xywh = []
    boxes_xyxy = []
    scores = []

    x_factor = w / float(imgsz)
    y_factor = h / float(imgsz)

    for row in out:
        if row.shape[0] < 6:
            continue

        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < conf:
            continue
        if classes is not None and len(classes) > 0 and class_id not in classes:
            continue

        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        x1 = int((cx - bw / 2.0) * x_factor)
        y1 = int((cy - bh / 2.0) * y_factor)
        x2 = int((cx + bw / 2.0) * x_factor)
        y2 = int((cy + bh / 2.0) * y_factor)
        clamped = clamp_box(x1, y1, x2, y2, w, h)
        if clamped is None:
            continue
        x1c, y1c, x2c, y2c = clamped

        boxes_xyxy.append((x1c, y1c, x2c, y2c))
        boxes_xywh.append([x1c, y1c, max(1, x2c - x1c), max(1, y2c - y1c)])
        scores.append(score)

    if not boxes_xywh:
        return []

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf, 0.45)
    if len(indices) == 0:
        return []

    selected = []
    for idx in np.array(indices).flatten().tolist():
        selected.append(boxes_xyxy[idx])
    return selected


def process_frame(frame, slots, detector, backend, classes, conf, imgsz, device, overlap_threshold=0.15):
    if backend == "ultralytics":
        detected_boxes = detect_with_ultralytics(
            frame, detector, classes, conf, imgsz, device)
    else:
        detected_boxes = detect_with_opencv_onnx(
            frame, detector, classes, conf, imgsz)

    occupied_indices = set()
    for db in detected_boxes:
        best_idx = -1
        best_score = 0.0
        best_slot_area = None
        db_area = rect_area(db)
        if db_area <= 0:
            continue
        for i, s in enumerate(slots):
            inter = intersection_area(db, s)
            if inter <= 0:
                continue
            score = inter / float(db_area)
            if score < overlap_threshold:
                continue

            slot_area = rect_area(s)
            if best_slot_area is None:
                best_slot_area = slot_area
                best_score = score
                best_idx = i
                continue

            # Prefer the smallest valid overlapping slot. This avoids assigning
            # all objects to one giant container slot if slot annotations include one.
            if slot_area < best_slot_area or (slot_area == best_slot_area and score > best_score):
                best_slot_area = slot_area
                best_score = score
                best_idx = i

        if best_idx >= 0:
            occupied_indices.add(best_idx)

    frame_area = float(frame.shape[0] * frame.shape[1])
    suspicious_detector_output = any((rect_area(
        db) / frame_area) > 0.55 for db in detected_boxes) if frame_area > 0 else False

    # Fallback when detector under-counts or emits a dominant frame-sized box.
    if (len(occupied_indices) <= 1 or suspicious_detector_output) and len(slots) >= 2:
        fallback_candidates = visual_occupied_slot_candidates(frame, slots)
        if len(fallback_candidates) > len(occupied_indices):
            occupied_indices = set(fallback_candidates)

    annotated = frame.copy()
    for i, s in enumerate(slots):
        occ = i in occupied_indices
        color = (0, 0, 255) if occ else (0, 255, 0)
        cv2.rectangle(annotated, (s[0], s[1]), (s[2], s[3]), color, 2)

    total = len(slots)
    occupied = len(occupied_indices)
    free = total - occupied
    cv2.putText(annotated, f"Free: {free} / {total}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return annotated, occupied, free, total


def main():
    p = argparse.ArgumentParser(description="Parking slot occupancy detector")
    p.add_argument("image", nargs="?", default=None,
                   help="Path to input image (optional when --camera is used)")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    p.add_argument("--backend", choices=["ultralytics", "opencv"], default="ultralytics",
                   help="Inference backend: ultralytics (PyTorch) or opencv (ONNX). Use opencv on Raspberry Pi if torch fails.")
    p.add_argument("--slots", default="annotated_parking_coords.txt",
                   help="Slots file (x1,y1,x2,y2[,flag])")
    p.add_argument("--grid-rows", type=int, default=0,
                   help="Generate slot grid rows dynamically (example: 3)")
    p.add_argument("--grid-cols", type=int, default=0,
                   help="Generate slot grid columns dynamically (example: 2)")
    p.add_argument("--grid-roi", type=str, default=None,
                   help="Optional ROI for grid as x1,y1,x2,y2; if omitted, whole image is used")
    p.add_argument("--output", "-o", default="images/parking_out.jpg",
                   help="Annotated output image")
    p.add_argument("--classes", nargs="*", type=int, default=None,
                   help="Class IDs to treat as occupied objects (default: all classes)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold for detections (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=416,
                   help="Inference image size for YOLO (default: 416, good for Raspberry Pi)")
    p.add_argument("--camera", action="store_true",
                   help="Use webcam/video capture mode instead of single image")
    p.add_argument("--camera-index", type=int, default=0,
                   help="Camera index for cv2.VideoCapture (default: 0)")
    p.add_argument("--show", action="store_true",
                   help="Display OpenCV window (off by default for headless Raspberry Pi)")
    p.add_argument("--device", default="cpu",
                   help="YOLO device, e.g. cpu or 0 (default: cpu)")
    p.add_argument("--save-interval", type=int, default=30,
                   help="In camera mode, save annotated frame every N frames (default: 30)")
    args = p.parse_args()

    if not args.camera:
        if not args.image:
            print("Error: image path is required when not using --camera")
            return
        if not os.path.exists(args.image):
            print("Error: image not found:", args.image)
            return

    if not os.path.exists(args.model):
        print("Error: YOLO model not found:", args.model)
        return

    if args.backend == "ultralytics":
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("Error: failed to import ultralytics.")
            print("This can happen on Raspberry Pi due to incompatible torch build.")
            print("Use ONNX + OpenCV backend instead:")
            print("python parking_system.py <image> --backend opencv --model yolo11n.onnx --slots annotated_parking_coords.txt -o images/out.jpg")
            print("Import error:", e)
            return
        detector = YOLO(args.model)
    else:
        if not args.model.lower().endswith(".onnx"):
            print(
                "Error: OpenCV backend requires an ONNX model file (example: yolo11n.onnx)")
            return
        try:
            detector = cv2.dnn.readNetFromONNX(args.model)
        except cv2.error as e:
            print("Error: failed to load ONNX model with OpenCV:", e)
            return

    if args.camera:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"Error: could not open camera index {args.camera_index}")
            return

        slots = load_slots(args.slots) if not (
            args.grid_rows > 0 and args.grid_cols > 0) else None
        roi = None
        if args.grid_roi:
            try:
                vals = [int(v.strip()) for v in args.grid_roi.split(",")]
                if len(vals) == 4:
                    roi = tuple(vals)
            except ValueError:
                roi = None
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from camera")
                break

            frame_count += 1
            if args.grid_rows > 0 and args.grid_cols > 0:
                scaled_slots = generate_grid_slots(
                    frame.shape[1], frame.shape[0], args.grid_rows, args.grid_cols, roi=roi
                )
            else:
                scaled_slots = scale_slots_to_image(
                    slots, frame.shape[1], frame.shape[0])
            annotated, occupied, free, total = process_frame(
                frame, scaled_slots, detector, args.backend, args.classes, args.conf, args.imgsz, args.device
            )

            if args.show:
                try:
                    cv2.imshow("Parking", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except cv2.error:
                    pass

            if frame_count % max(1, args.save_interval) == 0:
                output_dir = os.path.dirname(args.output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(args.output, annotated)
                print(
                    f"Frame {frame_count}: Total={total}, Occupied={occupied}, Free={free}")

        cap.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        return

    img = cv2.imread(args.image)
    if img is None:
        print("Error: failed to read image")
        return

    roi = None
    if args.grid_roi:
        try:
            vals = [int(v.strip()) for v in args.grid_roi.split(",")]
            if len(vals) == 4:
                roi = tuple(vals)
        except ValueError:
            roi = None

    if args.grid_rows > 0 and args.grid_cols > 0:
        scaled_slots = generate_grid_slots(
            img.shape[1], img.shape[0], args.grid_rows, args.grid_cols, roi=roi
        )
    else:
        slots = load_slots(args.slots)
        scaled_slots = scale_slots_to_image(slots, img.shape[1], img.shape[0])
    annotated, occupied, free, total = process_frame(
        img, scaled_slots, detector, args.backend, args.classes, args.conf, args.imgsz, args.device
    )

    if args.show:
        try:
            cv2.imshow("Parking", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, annotated)
    print("Saved annotated image to", args.output)
    print(f"Total slots: {total}; Occupied: {occupied}; Free: {free}")


if __name__ == '__main__':
    main()
