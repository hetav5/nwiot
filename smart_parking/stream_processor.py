import threading
import time
import os
from pathlib import Path
import cv2
import json
from ultralytics import YOLO

from .detector import parse_results
from .slots import load_slots, detection_inside_slot
from .transform import bbox_size_in_bev, get_birds_eye


class CameraStream:
    def __init__(self, source, model_path='yolov8m.pt', slots_json='data/slots_example.json', out_dir='runs/parking_stream', conf=0.25, pxpm=100.0, slot_real_size=(5.0, 2.5), frame_skip=5):
        self.source = source
        self.model_path = model_path
        self.slots_json = slots_json
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.conf = conf
        self.pxpm = pxpm
        self.slot_real_size = slot_real_size
        self.frame_skip = int(frame_skip)

        self._stop = threading.Event()
        self._thread = None

        self._latest_report_path = str(self.out_dir / 'report.json')
        self._latest_img_path = str(self.out_dir / 'annotated.jpg')

        self._cap = None
        self._model = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass

    def _init_camera_and_model(self):
        # Video capture
        self._cap = cv2.VideoCapture(self.source)
        # load model once
        self._model = YOLO(self.model_path)
        # load slots
        self._slots = load_slots(self.slots_json)

    def _run(self):
        try:
            self._init_camera_and_model()
        except Exception as e:
            return

        frame_idx = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_idx += 1
            if frame_idx % self.frame_skip != 0:
                continue

            try:
                results = self._model(frame)
                dets = parse_results(results)
            except Exception:
                dets = []

            report = []
            for slot in self._slots:
                name = slot.get('name')
                poly = slot.get('polygon')
                if not poly:
                    report.append({'slot': name, 'error': 'invalid polygon'})
                    continue

                occupied = False
                vehicle_info = None
                for d in dets:
                    if d.get('conf', 0.0) < self.conf:
                        continue
                    if detection_inside_slot(d['xyxy'], poly):
                        occupied = True
                        # compute size in BEV
                        dst_w = int(self.slot_real_size[0] * self.pxpm)
                        dst_h = int(self.slot_real_size[1] * self.pxpm)
                        _, M = get_birds_eye(
                            frame, src_pts=poly, dst_size=(dst_w, dst_h))
                        size = bbox_size_in_bev(
                            d['xyxy'], M, px_per_meter=self.pxpm, target_slot_size=self.slot_real_size)
                        vehicle_info = {
                            'class': int(d['class']),
                            'conf': float(d['conf']),
                            'length_m': float(size['length_m']),
                            'width_m': float(size['width_m']),
                            'is_too_big': bool(size['is_too_big'])
                        }
                        break

                report.append(
                    {'slot': name, 'occupied': occupied, 'vehicle': vehicle_info})

            # visualization: draw slots and detections
            vis = frame.copy()
            import numpy as np
            for r in report:
                slot = next(
                    (s for s in self._slots if s.get('name') == r['slot']), None)
                if not slot:
                    continue
                poly = slot.get('polygon')
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(vis, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                cx, cy = int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
                label = f"{r['slot']}: {'Occ' if r.get('occupied') else 'Free'}"
                cv2.putText(vis, label, (cx-30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for d in dets:
                x1, y1, x2, y2 = map(int, d['xyxy'])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, f"c:{d['class']} {d['conf']:.2f}",
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # save latest
            try:
                cv2.imwrite(self._latest_img_path, vis)
                with open(self._latest_report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
            except Exception:
                pass

        # cleanup
        try:
            self._cap.release()
        except Exception:
            pass
