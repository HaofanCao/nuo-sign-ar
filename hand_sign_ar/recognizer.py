from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np


GESTURE_CN = {
    "OPEN_PALM": "开掌·祈福",
    "SWORD_SIGN": "剑诀·镇邪",
    "PINCH_SEAL": "合印·守护",
    "FIST_GUARD": "握拳·护持",
    "UNKNOWN": "未识别",
}


@dataclass
class GestureResult:
    name: str
    confidence: float
    scores: Dict[str, float]


class HandGestureRecognizer:
    """Rule-based hand-sign recognizer using 21 hand landmarks.

    Input landmarks must be np.ndarray shape (21, 3), normalized to [0, 1].
    """

    def __init__(self, confidence_threshold: float = 0.55, score_margin: float = 0.06) -> None:
        self.confidence_threshold = confidence_threshold
        self.score_margin = score_margin

    @staticmethod
    def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _extract_features(self, lm: np.ndarray) -> Dict[str, float]:
        wrist = lm[0]
        palm_size = max(self._dist(lm[0], lm[9]), self._dist(lm[5], lm[17]), 1e-6)

        def finger_ext(tip: int, pip: int) -> float:
            # y-axis in image grows downward: tip_y < pip_y means more extended.
            return self._clamp((float(lm[pip][1] - lm[tip][1]) + 0.01) / 0.20)

        thumb_ext = self._clamp((self._dist(wrist, lm[4]) - self._dist(wrist, lm[2])) / (0.30 * palm_size))
        index_ext = finger_ext(8, 6)
        middle_ext = finger_ext(12, 10)
        ring_ext = finger_ext(16, 14)
        pinky_ext = finger_ext(20, 18)

        thumb_index = self._dist(lm[4], lm[8]) / palm_size
        index_middle = self._dist(lm[8], lm[12]) / palm_size
        thumb_to_index_mcp = self._dist(lm[4], lm[5]) / palm_size

        return {
            "thumb_ext": thumb_ext,
            "index_ext": index_ext,
            "middle_ext": middle_ext,
            "ring_ext": ring_ext,
            "pinky_ext": pinky_ext,
            "thumb_index": thumb_index,
            "index_middle": index_middle,
            "thumb_to_index_mcp": thumb_to_index_mcp,
        }

    def _score_gestures(self, f: Dict[str, float]) -> Dict[str, float]:
        spread = self._clamp((f["index_middle"] - 0.18) / 0.28)
        pinch_close = self._clamp(1.0 - (f["thumb_index"] - 0.11) / 0.34)
        thumb_tucked = self._clamp(1.0 - (f["thumb_to_index_mcp"] - 0.18) / 0.35)

        open_palm = (
            0.18 * f["thumb_ext"]
            + 0.20 * f["index_ext"]
            + 0.20 * f["middle_ext"]
            + 0.20 * f["ring_ext"]
            + 0.17 * f["pinky_ext"]
            + 0.05 * spread
        )

        sword_sign = (
            0.28 * f["index_ext"]
            + 0.28 * f["middle_ext"]
            + 0.16 * (1.0 - f["ring_ext"])
            + 0.16 * (1.0 - f["pinky_ext"])
            + 0.12 * thumb_tucked
        )

        pinch_seal = (
            0.46 * pinch_close
            + 0.20 * f["index_ext"]
            + 0.12 * (1.0 - f["middle_ext"])
            + 0.11 * (1.0 - f["ring_ext"])
            + 0.11 * (1.0 - f["pinky_ext"])
        )

        fist_guard = (
            0.22 * (1.0 - f["thumb_ext"])
            + 0.20 * (1.0 - f["index_ext"])
            + 0.20 * (1.0 - f["middle_ext"])
            + 0.20 * (1.0 - f["ring_ext"])
            + 0.18 * (1.0 - f["pinky_ext"])
        )

        return {
            "OPEN_PALM": self._clamp(open_palm),
            "SWORD_SIGN": self._clamp(sword_sign),
            "PINCH_SEAL": self._clamp(pinch_seal),
            "FIST_GUARD": self._clamp(fist_guard),
        }

    def classify(self, landmarks: Optional[np.ndarray]) -> GestureResult:
        if landmarks is None or landmarks.shape != (21, 3):
            return GestureResult("UNKNOWN", 0.0, {})

        features = self._extract_features(landmarks)
        scores = self._score_gestures(features)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score < self.confidence_threshold:
            return GestureResult("UNKNOWN", float(best_score), scores)
        if (best_score - second_score) < self.score_margin:
            return GestureResult("UNKNOWN", float(best_score), scores)
        return GestureResult(best_name, float(best_score), scores)


class TemporalGestureSmoother:
    """Weighted majority vote over recent predictions."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, int(window_size))
        self.buf: Deque[GestureResult] = deque(maxlen=self.window_size)

    def update(self, result: GestureResult) -> Tuple[str, float]:
        self.buf.append(result)
        weighted_sum: Dict[str, float] = defaultdict(float)
        total = 0.0

        n = len(self.buf)
        for i, item in enumerate(self.buf):
            # Recent predictions receive higher weight.
            w = 0.65 + 0.35 * (i + 1) / n
            s = max(0.0, min(1.0, item.confidence))
            weighted_sum[item.name] += w * s
            total += w * s

        if not weighted_sum:
            return "UNKNOWN", 0.0

        name, score = max(weighted_sum.items(), key=lambda x: x[1])
        stability = score / total if total > 1e-6 else 0.0
        return name, float(stability)


def landmarks_from_mediapipe(hand_landmarks: Iterable) -> np.ndarray:
    points = []
    for lm in hand_landmarks:
        points.append((float(lm.x), float(lm.y), float(lm.z)))
    arr = np.array(points, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected (21,3) landmarks, got {arr.shape}")
    return arr
