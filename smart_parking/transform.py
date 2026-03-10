import cv2
import numpy as np
from typing import Tuple, List, Union, Dict


def get_birds_eye(image: np.ndarray, src_pts: List[Tuple[float, float]], dst_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bird's-eye (top-down) view for a parking slot.

    Args:
        image: source image (HxWxC)
        src_pts: 4 source points on the image corresponding to the slot corners.
                 Order should correspond to top-left, top-right, bottom-right, bottom-left
                 (clockwise or consistent ordering).
        dst_size: (width, height) of the output warped image in pixels.

    Returns:
        warped: the warped (bird's-eye) image of size `dst_size`
        M: 3x3 perspective transform matrix mapping source->destination
    """
    dst_w, dst_h = dst_size
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array([[0.0, 0.0], [dst_w - 1.0, 0.0], [dst_w - 1.0, dst_h - 1.0], [0.0, dst_h - 1.0]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return warped, M


def bbox_size_in_bev(
    bbox: Tuple[float, float, float, float],
    M: np.ndarray,
    px_per_meter: float = 100.0,
    target_slot_size: Union[Tuple[float, float], float] = (5.0, 2.5),
) -> Dict[str, Union[float, bool, Tuple[float, float, float]]]:
    """Map a YOLO bbox to bird's-eye view and calculate vehicle size in meters.

    Args:
        bbox: (x1, y1, x2, y2) in image pixel coordinates.
        M: perspective transform matrix (source -> BEV) from `get_birds_eye`.
        px_per_meter: scale (pixels per meter) in BEV image. Default 100 px == 1 m.
        target_slot_size: (length_m, width_m) target allowed size in meters,
                          or a single scalar to compare the vehicle length against.

    Returns:
        dict with keys:
          - 'length_m': vehicle major dimension in meters
          - 'width_m': vehicle minor dimension in meters
          - 'is_too_big': boolean indicating if vehicle exceeds `target_slot_size`
          - 'rect': ( (cx,cy), (w_px, h_px), angle ) the min-area rect in BEV pixels
          - 'bev_pts': mapped corner points in BEV coordinates (4x2 array)
    """
    x1, y1, x2, y2 = bbox
    # source corners of bbox (clockwise)
    pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    bev_pts = cv2.perspectiveTransform(pts, M)[0]  # shape (4,2)

    # Use minAreaRect to get oriented bounding box in BEV coordinates (pixels)
    rect = cv2.minAreaRect(bev_pts.astype(np.float32))
    (_, _), (w_px, h_px), angle = rect

    # convert to meters
    w_m = w_px / px_per_meter
    h_m = h_px / px_per_meter
    length_m = max(w_m, h_m)
    width_m = min(w_m, h_m)

    # compare to target_slot_size
    if isinstance(target_slot_size, (list, tuple)) and len(target_slot_size) == 2:
        target_length, target_width = float(target_slot_size[0]), float(target_slot_size[1])
        is_too_big = (length_m > target_length) or (width_m > target_width)
    else:
        # scalar: compare major dimension only
        is_too_big = length_m > float(target_slot_size)

    return {
        "length_m": float(length_m),
        "width_m": float(width_m),
        "is_too_big": bool(is_too_big),
        "rect": rect,
        "bev_pts": bev_pts,
    }
