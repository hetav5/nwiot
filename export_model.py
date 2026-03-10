"""Export a trained YOLO model to other formats (onnx, torchscript, etc.).

Usage:
  python export_model.py --weights runs/detect/parking_train/weights/best.pt --format onnx
"""
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to weights (best.pt)")
    p.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "tflite", "coreml", "saved_model"], help="Export format")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.export(format=args.format)


if __name__ == "__main__":
    main()
