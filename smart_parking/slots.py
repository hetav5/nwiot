import json
from typing import List, Tuple, Dict

Point = Tuple[float, float]
Polygon = List[Point]


def save_slots(path: str, slots: List[Dict]) -> None:
    """Save slots (list of dicts with 'name' and 'polygon') to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=2)


def load_slots(path: str) -> List[Dict]:
    """Load slots JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
    return inside


def bbox_center(xyxy: Tuple[float, float, float, float]) -> Point:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detection_inside_slot(detection_bbox: Tuple[float, float, float, float], polygon: Polygon) -> bool:
    """Decide if detection (xyxy) is inside polygon by checking bbox center."""
    cx, cy = bbox_center(detection_bbox)
    return point_in_polygon((cx, cy), polygon)
