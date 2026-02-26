from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .recognizer import GestureResult, HandGestureRecognizer, TemporalGestureSmoother


@dataclass
class BenchmarkRow:
    config_id: str
    smoothing: int
    threshold: float
    margin: float
    accuracy: float
    false_alarm_rate: float
    avg_confidence: float
    latency_ms: float
    composite_score: float


def _base_hand() -> np.ndarray:
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.50, 0.86, 0.0]

    # Thumb skeleton defaults (open-ish)
    lm[1] = [0.44, 0.78, 0.0]
    lm[2] = [0.39, 0.72, 0.0]
    lm[3] = [0.34, 0.66, 0.0]
    lm[4] = [0.29, 0.60, 0.0]

    # Finger MCP anchors
    lm[5] = [0.43, 0.70, 0.0]
    lm[9] = [0.50, 0.69, 0.0]
    lm[13] = [0.57, 0.70, 0.0]
    lm[17] = [0.64, 0.72, 0.0]
    return lm


def _set_finger(lm: np.ndarray, base_idx: int, open_state: bool) -> None:
    x = lm[base_idx][0]
    y0 = lm[base_idx][1]
    if open_state:
        lm[base_idx + 1] = [x, y0 - 0.12, 0.0]
        lm[base_idx + 2] = [x, y0 - 0.22, 0.0]
        lm[base_idx + 3] = [x, y0 - 0.31, 0.0]
    else:
        lm[base_idx + 1] = [x + 0.01, y0 - 0.05, 0.0]
        lm[base_idx + 2] = [x + 0.02, y0 - 0.02, 0.0]
        lm[base_idx + 3] = [x + 0.02, y0 + 0.01, 0.0]


def canonical_landmarks(gesture: str) -> np.ndarray:
    lm = _base_hand()

    if gesture == "OPEN_PALM":
        thumb_open = True
        states = [True, True, True, True]  # index, middle, ring, pinky
    elif gesture == "SWORD_SIGN":
        thumb_open = False
        states = [True, True, False, False]
    elif gesture == "PINCH_SEAL":
        thumb_open = False
        states = [True, False, False, False]
    elif gesture == "FIST_GUARD":
        thumb_open = False
        states = [False, False, False, False]
    else:
        thumb_open = True
        states = [True, False, True, False]

    if thumb_open:
        lm[1] = [0.44, 0.78, 0.0]
        lm[2] = [0.38, 0.72, 0.0]
        lm[3] = [0.32, 0.65, 0.0]
        lm[4] = [0.26, 0.58, 0.0]
    else:
        lm[1] = [0.45, 0.78, 0.0]
        lm[2] = [0.43, 0.76, 0.0]
        lm[3] = [0.41, 0.74, 0.0]
        lm[4] = [0.40, 0.72, 0.0]

    for base_idx, st in zip([5, 9, 13, 17], states):
        _set_finger(lm, base_idx, st)

    if gesture == "PINCH_SEAL":
        # Move thumb tip close to index tip to emulate pinch.
        lm[8] = [0.44, 0.42, 0.0]
        lm[4] = [0.45, 0.44, 0.0]

    return lm


def noisy_sample(gesture: str, noise_std: float, rng: np.random.Generator) -> Optional[np.ndarray]:
    if gesture == "NO_HAND":
        return None
    lm = canonical_landmarks(gesture)
    noise = rng.normal(0.0, noise_std, size=lm.shape).astype(np.float32)
    lm = np.clip(lm + noise, 0.0, 1.0)
    return lm


def build_eval_stream() -> List[Tuple[str, Optional[np.ndarray]]]:
    rng = np.random.default_rng(20260226)
    stream: List[Tuple[str, Optional[np.ndarray]]] = []

    plan = [
        ("OPEN_PALM", 120, 0.010),
        ("SWORD_SIGN", 120, 0.014),
        ("PINCH_SEAL", 120, 0.015),
        ("FIST_GUARD", 120, 0.012),
        ("NO_HAND", 80, 0.0),
        ("OPEN_PALM", 80, 0.020),
        ("PINCH_SEAL", 80, 0.022),
        ("SWORD_SIGN", 80, 0.025),
    ]

    for gesture, frames, noise in plan:
        for _ in range(frames):
            sample = noisy_sample(gesture, noise, rng)
            label = "UNKNOWN" if gesture == "NO_HAND" else gesture
            stream.append((label, sample))

    return stream


def evaluate_config(smoothing: int, threshold: float, margin: float) -> BenchmarkRow:
    recognizer = HandGestureRecognizer(confidence_threshold=threshold, score_margin=margin)
    smoother = TemporalGestureSmoother(window_size=smoothing)
    stream = build_eval_stream()

    total = len(stream)
    correct = 0
    false_alarm = 0
    unknown_total = 0
    conf_sum = 0.0

    for label, sample in stream:
        raw = recognizer.classify(sample)
        pred, stability = smoother.update(raw)

        if pred == label:
            correct += 1

        if label == "UNKNOWN":
            unknown_total += 1
            if pred != "UNKNOWN":
                false_alarm += 1

        conf_sum += max(raw.confidence, 0.75 * stability)

    accuracy = correct / max(1, total)
    far = false_alarm / max(1, unknown_total)
    avg_conf = conf_sum / max(1, total)

    latency_ms = (smoothing - 1) / 30.0 * 1000.0
    composite = (
        accuracy * 100.0 * 0.65
        + (1.0 - far) * 100.0 * 0.20
        + avg_conf * 100.0 * 0.10
        + max(0.0, 100.0 - latency_ms * 0.45) * 0.05
    )

    cid = f"W{smoothing}_T{threshold:.2f}_M{margin:.2f}"
    return BenchmarkRow(
        config_id=cid,
        smoothing=smoothing,
        threshold=threshold,
        margin=margin,
        accuracy=accuracy,
        false_alarm_rate=far,
        avg_confidence=avg_conf,
        latency_ms=latency_ms,
        composite_score=composite,
    )


def run_grid_search(
    smoothing_list: Sequence[int] = (3, 5, 7),
    threshold_list: Sequence[float] = (0.50, 0.55, 0.60),
    margin_list: Sequence[float] = (0.04, 0.06, 0.08),
) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []
    for w, t, m in product(smoothing_list, threshold_list, margin_list):
        rows.append(evaluate_config(w, float(t), float(m)))
    rows.sort(key=lambda x: x.composite_score, reverse=True)
    return rows


def save_csv(rows: Sequence[BenchmarkRow], csv_path: Path) -> None:
    header = "config_id,smoothing,threshold,margin,accuracy,false_alarm_rate,avg_confidence,latency_ms,composite_score\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.config_id},{r.smoothing},{r.threshold:.2f},{r.margin:.2f},"
            f"{r.accuracy:.6f},{r.false_alarm_rate:.6f},{r.avg_confidence:.6f},"
            f"{r.latency_ms:.2f},{r.composite_score:.4f}\n"
        )
    csv_path.write_text("".join(lines), encoding="utf-8")


def save_chart(rows: Sequence[BenchmarkRow], png_path: Path, top_k: int = 8) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rcParams
    except Exception as exc:
        raise RuntimeError("matplotlib is required to draw parameter chart") from exc

    # Try to enable Chinese font on Windows for readable labels.
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]:
        p = Path(fp)
        if not p.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(p))
            font_name = font_manager.FontProperties(fname=str(p)).get_name()
            rcParams["font.family"] = font_name
            rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    top = list(rows[:top_k])
    labels = [r.config_id for r in top]
    acc = [r.accuracy * 100.0 for r in top]
    far = [r.false_alarm_rate * 100.0 for r in top]
    lat = [r.latency_ms for r in top]

    x = np.arange(len(top))
    width = 0.34

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=160)
    b1 = ax1.bar(x - width / 2, acc, width, label="Accuracy (%)")
    b2 = ax1.bar(x + width / 2, [100.0 - y for y in far], width, label="(100-FAR) (%)")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")

    ax2 = ax1.twinx()
    l1 = ax2.plot(x, lat, color="#d62728", marker="o", linewidth=2.0, label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_ylim(0, max(260, max(lat) * 1.25))

    ax1.set_title("AR手诀识别参数对比（算法匹配）")
    handles = [b1, b2, l1[0]]
    labels_legend = ["准确率", "抗误报能力(100-FAR)", "响应延迟"]
    ax1.legend(handles, labels_legend, loc="upper right")

    # Annotate top composite score
    best = top[0]
    ax1.text(
        0.02,
        0.95,
        f"Best: {best.config_id} | Composite={best.composite_score:.2f}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#999", alpha=0.85),
    )

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)
    plt.close(fig)
