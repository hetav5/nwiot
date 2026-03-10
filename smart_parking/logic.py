from typing import List, Dict
from .slots import detection_inside_slot


class ParkingManager:
    """Manage parking slots and classify occupancy."""

    def __init__(self, slots: List[Dict]):
        # slots: list of {"name": str, "polygon": [(x,y), ...]}
        self.slots = slots

    def classify(self, detections: List[Dict]) -> List[Dict]:
        """Classify each slot as 'Empty' or 'Occupied'.

        detections: list of {"xyxy": (x1,y1,x2,y2), "class": str, "conf": float}
        Returns list of {"name":..., "status": "Empty"|"Occupied", "by": detection or None}
        """
        results = []
        for slot in self.slots:
            name = slot.get("name")
            polygon = slot.get("polygon", [])
            occupied_by = None
            for det in detections:
                if detection_inside_slot(det["xyxy"], polygon):
                    occupied_by = det
                    break
            results.append({"name": name, "status": "Occupied" if occupied_by else "Empty", "by": occupied_by})
        return results
