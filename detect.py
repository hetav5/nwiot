import cv2
import os
import sys
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO detect on an image")
    p.add_argument("image", nargs="?", default="images/detect.jpeg",
                   help="Path to image (default: images/detect.jpeg)")
    p.add_argument("--output", "-o", default="images/annotated_detect.jpg",
                   help="Output image when GUI not available")
    return p.parse_args()


def check_overlap(slot, cars):
    sx1, sy1, sx2, sy2 = slot
    for car in cars:
        cx1, cy1, cx2, cy2 = car
        if not (cx2 < sx1 or cx1 > sx2 or cy2 < sy1 or cy1 > sy2):
            return True
    return False


def main():
    args = parse_args()
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        print("Pass a path: python detect.py path/to/image.jpg")
        sys.exit(1)

    # load model
    model = YOLO("yolo11n.pt")

    # load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: failed to read image: {image_path}")
        sys.exit(1)

    # resize for faster processing (preserve aspect by using max dim)
    image = cv2.resize(image, (640, 640))

    # parking slots (paste from annotation)
    parking_slots = [
        # (371, 0, 429, 25),
        (0, 28, 768, 236),
        (0, 236, 768, 444),
        (0, 444, 768, 652),
        (0, 652, 768, 860),
        (0, 860, 768, 1068),
        (0, 1068, 768, 1280)
    ]

    # run detection
    results = model(image)

    car_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            # car class id (YOLO class index may vary; adjust if needed)
            if cls == 2:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    free = 0
    for slot in parking_slots:
        x1, y1, x2, y2 = slot
        occupied = check_overlap(slot, car_boxes)
        color = (0, 0, 255) if occupied else (0, 255, 0)
        if not occupied:
            free += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    cv2.putText(image, f"Free Slots: {free}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Try to show image; if GUI unavailable, save to --output
    try:
        cv2.imshow("Parking Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print("OpenCV GUI not available; saving output to", args.output)
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(args.output, image)
        print("Saved:", args.output)


if __name__ == '__main__':
    main()
