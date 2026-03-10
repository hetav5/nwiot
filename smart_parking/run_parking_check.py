import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from smart_parking.detector import parse_results
from smart_parking.slots import load_slots, detection_inside_slot
from smart_parking.transform import get_birds_eye, bbox_size_in_bev


def check_image(image_path: str, model_path: str, slots_json: str, conf_thresh: float = 0.25, bev_px_per_meter: float = 100.0, slot_real_size=(5.0, 2.5)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Image not found or cannot be read: {image_path}")

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf_thresh, save=False)
    dets = parse_results(results)

    slots = load_slots(slots_json)
    report = []

    for slot in slots:
        slot_name = slot.get("name")
        polygon = slot.get("polygon")
        if not polygon or len(polygon) < 4:
            report.append({"slot": slot_name, "error": "invalid polygon"})
            continue

        # build BEV transform for this slot
        dst_w = int(slot_real_size[0] * bev_px_per_meter)
        dst_h = int(slot_real_size[1] * bev_px_per_meter)
        _, M = get_birds_eye(img, src_pts=polygon, dst_size=(dst_w, dst_h))

        occupied = False
        vehicle_info = None
        for d in dets:
            if d["conf"] < conf_thresh:
                continue
            if detection_inside_slot(d["xyxy"], polygon):
                occupied = True
                size = bbox_size_in_bev(
                    d["xyxy"], M, px_per_meter=bev_px_per_meter, target_slot_size=slot_real_size)
                vehicle_info = {
                    "class": int(d["class"]),
                    "conf": float(d["conf"]),
                    "length_m": float(size["length_m"]),
                    "width_m": float(size["width_m"]),
                    "is_too_big": bool(size["is_too_big"]),
                }
                break

        report.append({"slot": slot_name, "occupied": occupied,
                      "vehicle": vehicle_info})

    return report, dets


def visualize_and_save(image_path: str, slots_json: str, report, dets, out_dir: str):
    img = cv2.imread(image_path)
    slots = load_slots(slots_json)
    vis = img.copy()

    # draw slots
    for r in report:
        slot = next((s for s in slots if s.get("name") == r["slot"]), None)
        if not slot:
            continue
        poly = slot.get("polygon")
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        label = f"{r['slot']}: {'Occ' if r.get('occupied') else 'Free'}"
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        cv2.putText(vis, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # draw detections
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, f"c:{d['class']} {d['conf']:.2f}", (x1,
                    y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, "annotated.jpg")
    cv2.imwrite(out_img, vis)
    out_json = os.path.join(out_dir, "report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_img, out_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Image path to check")
    p.add_argument("--model", default="yolov8m.pt", help="YOLO weights path")
    p.add_argument("--slots", default="data/slots_example.json",
                   help="Slots JSON path")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--pxpm", type=float, default=100.0,
                   help="BEV pixels per meter")
    p.add_argument("--out", default="runs/parking_check",
                   help="Output directory for visuals and report")
    args = p.parse_args()

    report, dets = check_image(
        args.image, args.model, args.slots, conf_thresh=args.conf, bev_px_per_meter=args.pxpm)
    out_img, out_json = visualize_and_save(
        args.image, args.slots, report, dets, args.out)
    print("Report:")
    print(json.dumps(report, indent=2))
    print(f"Saved annotated image: {out_img}")
    print(f"Saved JSON report: {out_json}")


if __name__ == "__main__":
    main()
