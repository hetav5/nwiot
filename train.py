"""Train YOLOv8 model on `data/vehicles.yaml`.

Usage:
  python train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt", help="Base model or weights")
    p.add_argument("--data", default="data/vehicles.yaml", help="Dataset YAML")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", type=str, default="parking_train")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)


if __name__ == "__main__":
    main()
