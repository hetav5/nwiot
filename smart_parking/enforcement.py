import os
import csv
from datetime import datetime
from typing import Tuple, Optional

import cv2

try:
    import easyocr
except Exception:
    easyocr = None


def crop_bbox(frame: 'np.ndarray', bbox: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    return frame[y1:y2, x1:x2]


def read_license_plate_from_crop(crop_image, langs: Optional[list] = None, gpu: bool = False) -> str:
    """Run EasyOCR on a cropped vehicle image to attempt reading a plate.

    Returns the best text found or empty string.
    """
    if easyocr is None:
        raise RuntimeError("easyocr is not installed. Please install with `pip install easyocr`.")
    if langs is None:
        langs = ["en"]
    reader = easyocr.Reader(langs, gpu=gpu)
    # easyocr expects RGB images
    img_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)
    if not results:
        return ""
    # Choose result with highest confidence
    best = max(results, key=lambda r: r[2])
    return best[1]


def report_violation(plate_text: str, timestamp: datetime, snapshot_image, slot_id: str, csv_path: str = "police_report.csv"):
    """Save violation info to CSV and snapshot to disk, then print warning.

    CSV columns: timestamp, plate, slot_id, snapshot_path
    """
    os.makedirs("snapshots", exist_ok=True)
    if isinstance(timestamp, datetime):
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        iso_ts = timestamp.isoformat()
    else:
        ts_str = str(int(datetime.now().timestamp()))
        iso_ts = str(timestamp)

    safe_plate = plate_text.strip().replace(" ", "_") if plate_text else "UNKNOWN"
    fname = f"snapshots/{ts_str}_{safe_plate}.jpg"
    # save snapshot image (BGR)
    cv2.imwrite(fname, snapshot_image)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "plate", "slot_id", "snapshot_path"])
        writer.writerow([iso_ts, plate_text, slot_id, fname])

    print(f"Violation Detected: Plate [{plate_text}] is too large for Slot [{slot_id}]")


def enforce_if_too_big(slot_category: str, is_too_big: bool, bbox: Tuple[int, int, int, int], frame, slot_id: str, timestamp: Optional[datetime] = None, csv_path: str = "police_report.csv") -> bool:
    """If violation condition meets (too big and slot is 'Small'), crop, OCR plate, and report.

    Returns True if a violation was reported.
    """
    if not is_too_big:
        return False
    if slot_category.lower() != "small":
        return False

    crop = crop_bbox(frame, bbox)
    try:
        plate = read_license_plate_from_crop(crop)
    except RuntimeError:
        plate = ""

    if timestamp is None:
        timestamp = datetime.now()

    report_violation(plate, timestamp, crop, slot_id, csv_path=csv_path)
    return True
