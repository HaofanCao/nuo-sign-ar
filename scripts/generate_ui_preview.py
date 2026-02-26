from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_sign_ar.benchmark import canonical_landmarks
from hand_sign_ar.overlay import draw_ar_overlay


def make_bg(h: int, w: int) -> np.ndarray:
    y = np.linspace(0, 1, h).reshape(-1, 1)
    x = np.linspace(0, 1, w).reshape(1, -1)
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :, 0] = (35 + 40 * (1 - y)).astype(np.uint8)
    bg[:, :, 1] = (45 + 60 * x).astype(np.uint8)
    bg[:, :, 2] = (55 + 80 * (0.6 * x + 0.4 * y)).astype(np.uint8)
    cv2.rectangle(bg, (60, 120), (w - 70, h - 70), (30, 36, 46), -1)
    cv2.rectangle(bg, (60, 120), (w - 70, h - 70), (90, 100, 120), 1)
    cv2.putText(bg, "Simulated Camera Frame", (90, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (190, 200, 220), 2)
    return bg


def main() -> None:
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    h, w = 720, 1280
    frame = make_bg(h, w)

    lm = canonical_landmarks("SWORD_SIGN")
    frame = draw_ar_overlay(frame, lm, "SWORD_SIGN", 0.89, 30.1, recording=True)
    cv2.putText(frame, "This is a simulated preview frame", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 240), 2)

    out_file = out / "ar_ui_preview.png"
    ok, buf = cv2.imencode(".png", frame)
    if ok:
        buf.tofile(str(out_file))
        print(f"[OK] preview -> {out_file}")
    else:
        print("[warn] preview save failed")


if __name__ == "__main__":
    main()
