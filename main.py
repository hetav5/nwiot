"""Main runner for Smart Parking System.

Usage:
  python main.py --video PATH_TO_VIDEO --slots data/slots_example.json
"""
import argparse
import time
import json
import cv2
from ultralytics import YOLO
from smart_parking.slots import load_slots
from smart_parking.logic import ParkingManager
from smart_parking.detector import parse_results, filter_vehicle_classes


def draw_polygon(frame, polygon, color=(0, 255, 0), thickness=2):
    pts = [(int(x), int(y)) for x, y in polygon]
    if pts:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=0, help="Video file path or camera index")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path or name")
    parser.add_argument("--slots", default="data/slots_example.json", help="Slots JSON file")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    args = parser.parse_args()

    # Load slots
    slots = load_slots(args.slots)
    manager = ParkingManager(slots)

    # Load model
    model = YOLO(args.model)
    names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else model.names
    vehicle_idxs = filter_vehicle_classes(names)

    # Open video
    src = int(args.video) if str(args.video).isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to open video source")
        return

    import numpy as np

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf)
        detections = parse_results(results)
        # Filter to vehicle classes
        detections = [d for d in detections if d["class"] in vehicle_idxs]

        # Classify slots
        statuses = manager.classify(detections)

        # Draw slots and statuses
        for slot in slots:
            poly = slot.get("polygon", [])
            pts = np.array(poly, dtype=np.int32)
            color = (0, 255, 0)
            # find status for this slot
            s = next((st for st in statuses if st["name"] == slot.get("name")), None)
            if s and s["status"] == "Occupied":
                color = (0, 0, 255)
            if pts.size:
                cv2.polylines(frame, [pts], True, color, 2)
                if s:
                    # place label
                    x, y = pts[0]
                    cv2.putText(frame, f"{s['name']}: {s['status']}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw detections
        for d in detections:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        cv2.imshow("Smart Parking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
