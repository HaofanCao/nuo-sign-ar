from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .recognizer import GESTURE_CN

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    PIL_OK = True
except Exception:
    PIL_OK = False


COLOR_MAP = {
    "OPEN_PALM": (80, 210, 120),
    "SWORD_SIGN": (70, 170, 255),
    "PINCH_SEAL": (210, 150, 80),
    "FIST_GUARD": (200, 90, 220),
    "UNKNOWN": (160, 160, 160),
}


def _draw_text(frame: np.ndarray, text: str, org: Tuple[int, int], color=(230, 240, 255), size: int = 22) -> None:
    if not PIL_OK:
        cv2.putText(frame, text.encode("ascii", "ignore").decode(), org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return

    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    font = None
    for f in font_candidates:
        if Path(f).exists():
            try:
                font = ImageFont.truetype(f, size=size)
                break
            except Exception:
                pass
    if font is None:
        cv2.putText(frame, text.encode("ascii", "ignore").decode(), org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.text(org, text, fill=(int(color[2]), int(color[1]), int(color[0])), font=font)
    frame[:, :, :] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _blend_panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.35) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 25, 35), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 120, 170), 1)


def draw_ar_overlay(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    gesture_name: str,
    confidence: float,
    fps: float,
    recording: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    color = COLOR_MAP.get(gesture_name, COLOR_MAP["UNKNOWN"])

    if landmarks is not None and landmarks.shape == (21, 3):
        pts = np.column_stack((landmarks[:, 0] * w, landmarks[:, 1] * h)).astype(np.int32)

        palm_ids = [0, 5, 9, 13, 17]
        palm = pts[palm_ids]
        center = tuple(np.mean(palm, axis=0).astype(np.int32).tolist())

        radius = int(max(36, np.linalg.norm(pts[0] - pts[9]) * 0.9))
        overlay = frame.copy()

        cv2.circle(overlay, center, radius, color, 3)
        cv2.circle(overlay, center, int(radius * 1.28), color, 1)
        cv2.polylines(overlay, [palm.reshape(-1, 1, 2)], True, color, 2)

        # Tech-style reticle
        cv2.line(overlay, (center[0] - radius - 14, center[1]), (center[0] - radius + 6, center[1]), color, 2)
        cv2.line(overlay, (center[0] + radius - 6, center[1]), (center[0] + radius + 14, center[1]), color, 2)
        cv2.line(overlay, (center[0], center[1] - radius - 14), (center[0], center[1] - radius + 6), color, 2)
        cv2.line(overlay, (center[0], center[1] + radius - 6), (center[0], center[1] + radius + 14), color, 2)

        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    _blend_panel(frame, 18, 16, 500, 124, alpha=0.42)
    cn_name = GESTURE_CN.get(gesture_name, GESTURE_CN["UNKNOWN"])
    _draw_text(frame, f"手诀识别: {cn_name}", (34, 38), color=(230, 240, 255), size=24)
    _draw_text(frame, f"Gesture: {gesture_name}  |  Confidence: {confidence:.2f}", (34, 74), color=(180, 214, 255), size=18)
    _draw_text(frame, f"FPS: {fps:.1f}  |  Recording: {'ON' if recording else 'OFF'}", (34, 106), color=(170, 200, 230), size=18)

    if recording:
        cv2.circle(frame, (w - 34, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 74, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    return frame
