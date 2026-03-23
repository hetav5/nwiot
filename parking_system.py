import cv2
import os
import argparse
from ultralytics import YOLO


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


def main():
    p = argparse.ArgumentParser(description="Parking slot occupancy detector")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    p.add_argument("--slots", default="annotated_parking_coords.txt",
                   help="Slots file (x1,y1,x2,y2[,flag])")
    p.add_argument("--output", "-o", default="parking_out.jpg",
                   help="Annotated output image")
    p.add_argument("--classes", nargs="*", type=int, default=None,
                   help="Class IDs to treat as occupied objects (default: all classes)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold for detections (default: 0.25)")
    args = p.parse_args()

    if not os.path.exists(args.image):
        print("Error: image not found:", args.image)
        return

    if not os.path.exists(args.model):
        print("Error: YOLO model not found:", args.model)
        return

    img = cv2.imread(args.image)
    if img is None:
        print("Error: failed to read image")
        return

    slots = load_slots(args.slots)
    # scale slots if annotation coordinates were for a different resolution
    slots = scale_slots_to_image(slots, img.shape[1], img.shape[0])

    model = YOLO(args.model)
    results = model(img)

    detected_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf < args.conf:
                continue
            if args.classes is not None and len(args.classes) > 0 and cls not in args.classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_boxes.append((x1, y1, x2, y2))

    # Map each detected object to exactly one best-matching slot.
    occupied_indices = set()
    for db in detected_boxes:
        best_idx = -1
        best_score = 0.0
        db_area = rect_area(db)
        if db_area <= 0:
            continue
        for i, s in enumerate(slots):
            inter = intersection_area(db, s)
            if inter <= 0:
                continue
            score = inter / float(db_area)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= 0.15:
            occupied_indices.add(best_idx)

    annotated = img.copy()
    for i, s in enumerate(slots):
        occ = i in occupied_indices
        color = (0, 0, 255) if occ else (0, 255, 0)
        cv2.rectangle(annotated, (s[0], s[1]), (s[2], s[3]), color, 2)

    total = len(slots)
    occupied = len(occupied_indices)
    free = total - occupied

    cv2.putText(annotated, f"Free: {free} / {total}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    try:
        cv2.imshow("Parking", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        cv2.imwrite(args.output, annotated)
        print("Saved annotated image to", args.output)

    print(f"Total slots: {total}; Occupied: {occupied}; Free: {free}")


if __name__ == '__main__':
    main()
