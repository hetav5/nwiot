"""Run inference using a trained YOLO model on images or video.

Usage:
  python infer.py --weights runs/detect/parking_train/weights/best.pt --source tests/sample_video.mp4 --conf 0.25
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default=0, help="Image, directory, or video path (or camera index)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--save", action="store_true", help="Save annotated outputs")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=args.save)
    print("Inference done. Results object list length:", len(results))


if __name__ == "__main__":
    main()
