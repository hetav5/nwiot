from typing import List, Dict
import numpy as np


def filter_vehicle_classes(names_map):
    vehicle_keywords = {"car", "truck", "bus", "motor", "bike", "bicycle"}
    vehicle_idxs = [i for i, n in names_map.items() if any(k in n.lower() for k in vehicle_keywords)]
    return set(vehicle_idxs)


def parse_results(results) -> List[Dict]:
    """Convert ultralytics Results to simple detections list.

    Each detection is {"xyxy": (x1,y1,x2,y2), "class": name, "conf": float}
    """
    detections = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        xyxy_arr = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        cls_arr = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
        conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        for xyxy, cls, conf in zip(xyxy_arr, cls_arr, conf_arr):
            detections.append({"xyxy": tuple(map(float, xyxy)), "class": int(cls), "conf": float(conf)})
    return detections
