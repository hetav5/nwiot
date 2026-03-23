import cv2
import os
import sys
import argparse

slots = []
drawing = False
ix, iy = -1, -1


def draw(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        slots.append((ix, iy, x, y))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(
        description="Draw parking slots on an image interactively.")
    parser.add_argument("image", nargs="?", default="detect.jpeg",
                        help="Path to image (default: detect.jpeg)")
    parser.add_argument("--output", "-o", default="annotated_parking.jpg",
                        help="Output image when GUI not available")
    parser.add_argument("--expected-slots", type=int, default=0,
                        help="Expected number of parking slots (helps tuning detection)")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        print("Place the image in the script folder or pass a path: python draw_slots.py path/to/image.jpg")
        sys.exit(1)

    global img
    img = cv2.imread(image_path)
    if img is None:
        print(
            f"Error: failed to read image (corrupt or unsupported): {image_path}")
        sys.exit(1)

    # Try to use OpenCV GUI. If not available (headless build), fallback to automatic detection.
    try:
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw)

        while True:
            cv2.imshow("image", img)
            if cv2.waitKey(1) == 27:
                break

        print("Slots:", slots)
        cv2.destroyAllWindows()

    except cv2.error as e:
        print("OpenCV GUI not available in this build.\nDetails:", e)
        print("Falling back to automatic slot detection and saving output to", args.output)
        try:
            auto_slots = auto_detect_slots(img, args.expected_slots)

            # Try to detect cars using YOLO if model is available
            car_boxes = []
            model_path = "yolo11n.pt"
            if os.path.exists(model_path):
                try:
                    from ultralytics import YOLO
                    model = YOLO(model_path)
                    results = model(img)
                    for r in results:
                        for box in r.boxes:
                            if int(box.cls) == 2:  # car class id (adjust if needed)
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                car_boxes.append((x1, y1, x2, y2))
                except Exception as mex:
                    print("Warning: failed to run YOLO model:", mex)

            def check_overlap(slot, cars):
                sx1, sy1, sx2, sy2 = slot
                for car in cars:
                    cx1, cy1, cx2, cy2 = car
                    if not (cx2 < sx1 or cx1 > sx2 or cy2 < sy1 or cy1 > sy2):
                        return True
                return False

            free = 0
            annotated_slots = []
            for (x1, y1, x2, y2) in auto_slots:
                occupied = check_overlap((x1, y1, x2, y2), car_boxes)
                color = (0, 0, 255) if occupied else (0, 255, 0)
                if not occupied:
                    free += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                annotated_slots.append((x1, y1, x2, y2, int(occupied)))

            cv2.putText(img, f"Free Slots: {free}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imwrite(args.output, img)
            print("Saved:", args.output)
            print(f"Detected slots: {len(annotated_slots)}; Free: {free}")

            # Save coordinates with occupancy flag
            coords_path = os.path.splitext(args.output)[0] + "_coords.txt"
            with open(coords_path, "w") as f:
                for r in annotated_slots:
                    f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
            print("Coordinates saved to", coords_path)
        except Exception as ex:
            print("Automatic detection failed:", ex)


def auto_detect_slots(image, expected=0):
    """Improved automatic slot detection with multi-pass heuristics.

    Returns list of rectangles (x1,y1,x2,y2). Use `expected` to relax rules when
    fewer rectangles are found than expected.
    """
    def detect_by_threshold(img, blur_ksize=(5, 5), adapt=True, canny=False, morph_k=(7, 7)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, blur_ksize, 0)

        if adapt:
            proc = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 2)
        elif canny:
            proc = cv2.Canny(blur, 50, 150)
        else:
            _, proc = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_k)
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel, iterations=2)
        proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects_local = []
        h, w = img.shape[:2]
        img_area = w * h

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(500, img_area * 0.0008):
                continue

            x, y, rw, rh = cv2.boundingRect(cnt)
            rect_area = rw * rh
            solidity = area / rect_area if rect_area > 0 else 0
            ar = rw / float(rh) if rh > 0 else 0

            if rw > 30 and rh > 20 and 0.2 < ar < 6.0 and solidity > 0.3:
                rects_local.append((x, y, x + rw, y + rh))

        return rects_local

    # attempt 1: adaptive threshold (strict)
    rects = detect_by_threshold(image, blur_ksize=(
        5, 5), adapt=True, canny=False, morph_k=(9, 9))

    # attempt 2: canny if nothing found
    if len(rects) == 0:
        rects = detect_by_threshold(image, blur_ksize=(
            3, 3), adapt=False, canny=True, morph_k=(7, 7))

    # if user provided expected slots and detection undercounts, relax parameters
    if expected > 0 and len(rects) < expected:
        more = detect_by_threshold(image, blur_ksize=(
            3, 3), adapt=True, canny=False, morph_k=(5, 5))
        rects.extend(more)

        # aggressive pass: lower area and solidity thresholds to find faint/thin slots
        more2 = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        proc = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(
            proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        img_area = w * h
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(200, img_area * 0.0002):
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            rect_area = rw * rh
            solidity = area / rect_area if rect_area > 0 else 0
            ar = rw / float(rh) if rh > 0 else 0
            if rw > 20 and rh > 12 and 0.15 < ar < 8.0 and solidity > 0.08:
                more2.append((x, y, x + rw, y + rh))
        rects.extend(more2)

    # simple NMS to remove overlapping boxes
    def rects_iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def nms(rects, iou_thresh=0.35):
        boxes = [list(r) for r in rects]
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        picked = []
        while boxes:
            a = boxes.pop(0)
            picked.append(tuple(a))
            boxes = [b for b in boxes if rects_iou(a, b) <= iou_thresh]
        return picked

    rects = nms(rects)
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    # If we still have fewer than expected slots, try splitting the largest detected
    if expected > 0 and len(rects) > 0 and len(rects) < expected:
        def split_rect(rect, k):
            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1
            parts = []
            if w >= h:
                part_w = w // k
                for i in range(k):
                    sx = x1 + i * part_w
                    ex = x1 + (i + 1) * part_w if i < k - 1 else x2
                    parts.append((sx, y1, ex, y2))
            else:
                part_h = h // k
                for i in range(k):
                    sy = y1 + i * part_h
                    ey = y1 + (i + 1) * part_h if i < k - 1 else y2
                    parts.append((x1, sy, x2, ey))
            return parts

        largest = max(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        rects = [r for r in rects if r != largest]
        rects.extend(split_rect(largest, expected))
        rects = nms(rects)
        rects = sorted(rects, key=lambda r: (r[1], r[0]))
    return rects


if __name__ == '__main__':
    main()
