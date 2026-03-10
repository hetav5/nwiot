"""Validate a trained YOLO model.

Usage:
  python val.py --weights runs/detect/parking_train/weights/best.pt --data data/vehicles.yaml
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to weights (best.pt)")
    p.add_argument("--data", default="data/vehicles.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print(metrics)


if __name__ == "__main__":
    main()
