from __future__ import annotations

import argparse
import datetime as dt
import shutil
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from hand_sign_ar.overlay import draw_ar_overlay
from hand_sign_ar.recognizer import GestureResult, HandGestureRecognizer, TemporalGestureSmoother, landmarks_from_mediapipe

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
ASCII_MODEL_FALLBACK = Path("C:/AR_demo_models/hand_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AR hand-sign recognition demo for 傩戏手诀场景")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min-detect", type=float, default=0.65, help="MediaPipe detection confidence")
    parser.add_argument("--min-track", type=float, default=0.60, help="MediaPipe tracking confidence")
    parser.add_argument("--smoothing", type=int, default=5, help="temporal smoothing window")
    parser.add_argument("--threshold", type=float, default=0.55, help="gesture confidence threshold")
    parser.add_argument("--margin", type=float, default=0.06, help="gesture score margin")
    parser.add_argument("--output", default="outputs", help="output folder for screenshots and recordings")
    parser.add_argument("--backend", choices=["auto", "tasks", "solutions"], default="auto")
    parser.add_argument(
        "--model",
        default="C:/AR_demo_models/hand_landmarker.task",
        help="task model path for mediapipe tasks backend",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> tuple[Path, Path]:
    screenshots = base / "screenshots"
    recordings = base / "recordings"
    screenshots.mkdir(parents=True, exist_ok=True)
    recordings.mkdir(parents=True, exist_ok=True)
    return screenshots, recordings


def safe_imwrite(path: Path, frame: np.ndarray) -> bool:
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, frame)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def build_writer(path: Path, frame: np.ndarray, fps: float = 24.0) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (frame.shape[1], frame.shape[0]))


def build_ascii_temp_video_path() -> Path:
    tmp_root = Path(tempfile.gettempdir()) / "ar_hand_sign_demo"
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root / f"ar_demo_{uuid.uuid4().hex[:12]}.mp4"


def ensure_task_model(model_path: Path) -> Path:
    model_path = model_path.resolve()
    # MediaPipe task runtime on Windows can fail with non-ASCII model paths.
    if not str(model_path).isascii():
        model_path = ASCII_MODEL_FALLBACK
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print(f"[info] downloading hand model -> {model_path}")
    urllib.request.urlretrieve(MODEL_URL, str(model_path))
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError("model download failed")
    return model_path


def draw_task_landmarks(frame: np.ndarray, landmarks: List, connections: List[Tuple[int, int]]) -> None:
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        x = int(float(lm.x) * w)
        y = int(float(lm.y) * h)
        points.append((x, y))

    for s, e in connections:
        if s < len(points) and e < len(points):
            cv2.line(frame, points[s], points[e], (100, 220, 180), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 3, (245, 245, 245), -1)
        cv2.circle(frame, (x, y), 5, (60, 160, 255), 1)


def pick_backend(args: argparse.Namespace, mp) -> str:
    has_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    if args.backend == "solutions":
        if not has_solutions:
            raise RuntimeError("Requested --backend solutions, but current mediapipe has no solutions API.")
        return "solutions"
    if args.backend == "tasks":
        return "tasks"
    # auto
    return "solutions" if has_solutions else "tasks"


def main() -> None:
    args = parse_args()

    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("mediapipe is required. Please install from requirements.txt") from exc

    backend = pick_backend(args, mp)
    print(f"[info] mediapipe backend: {backend}")

    output_base = Path(args.output).resolve()
    ss_dir, rec_dir = ensure_dirs(output_base)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera. Check --camera index.")

    recognizer = HandGestureRecognizer(confidence_threshold=args.threshold, score_margin=args.margin)
    smoother = TemporalGestureSmoother(window_size=args.smoothing)

    recording = False
    writer = None
    record_target_path: Optional[Path] = None
    record_temp_path: Optional[Path] = None
    last_time = time.perf_counter()
    frame_idx = 0

    # backend objects
    hands = None
    mp_hands = None
    mp_draw = None
    mp_style = None
    task_landmarker = None
    task_connections: List[Tuple[int, int]] = []

    if backend == "solutions":
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        mp_style = mp.solutions.drawing_styles
        hands = mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=args.min_detect,
            min_tracking_confidence=args.min_track,
        )
    else:
        import mediapipe.tasks as mp_tasks

        model_path = ensure_task_model(Path(args.model))
        options = mp_tasks.vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=args.min_detect,
            min_hand_presence_confidence=max(0.5, args.min_track - 0.05),
            min_tracking_confidence=args.min_track,
        )
        task_landmarker = mp_tasks.vision.HandLandmarker.create_from_options(options)
        task_connections = [
            (int(c.start), int(c.end))
            for c in mp_tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        ]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            raw_result = GestureResult("UNKNOWN", 0.0, {})
            landmarks_arr: Optional[np.ndarray] = None

            if backend == "solutions":
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    landmarks_arr = landmarks_from_mediapipe(hand_lms.landmark)
                    raw_result = recognizer.classify(landmarks_arr)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )
            else:
                timestamp_ms = int(frame_idx * (1000.0 / 30.0))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = task_landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.hand_landmarks:
                    hand_lms = result.hand_landmarks[0]
                    landmarks_arr = landmarks_from_mediapipe(hand_lms)
                    raw_result = recognizer.classify(landmarks_arr)
                    draw_task_landmarks(frame, hand_lms, task_connections)

            smoothed_name, stability = smoother.update(raw_result)
            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - last_time)
            last_time = now
            frame_idx += 1

            conf = raw_result.confidence
            if smoothed_name != "UNKNOWN":
                conf = max(conf, 0.75 * stability + 0.25 * conf)
            else:
                conf = max(conf, stability * 0.6)

            frame = draw_ar_overlay(frame, landmarks_arr, smoothed_name, conf, fps, recording)
            cv2.putText(
                frame,
                "[Q] Quit  [S] Screenshot  [R] Record ON/OFF",
                (20, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (210, 220, 235),
                2,
            )

            if recording and writer is not None:
                writer.write(frame)

            cv2.imshow("AR Hand Sign Demo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("s"):
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                out = ss_dir / f"ar_demo_{ts}.png"
                if safe_imwrite(out, frame):
                    print(f"[saved] screenshot -> {out}")
                else:
                    print(f"[warn] screenshot failed: {out}")
            if key == ord("r"):
                recording = not recording
                if recording and writer is None:
                    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = rec_dir / f"ar_demo_{ts}.mp4"
                    record_target_path = out
                    record_temp_path = None
                    writer = build_writer(out, frame, fps=24.0)
                    if not writer.isOpened():
                        # Fallback for OpenCV Unicode-path issues:
                        # write to ASCII temp path first, then move back on stop.
                        tmp_out = build_ascii_temp_video_path()
                        record_temp_path = tmp_out
                        writer = build_writer(tmp_out, frame, fps=24.0)
                    if writer.isOpened():
                        shown = record_target_path if record_target_path else out
                        print(f"[recording] start -> {shown}")
                    else:
                        recording = False
                        writer.release()
                        writer = None
                        record_target_path = None
                        record_temp_path = None
                        print("[warn] recording start failed. Check write permissions and codec support.")
                elif not recording and writer is not None:
                    writer.release()
                    writer = None
                    # If recorded to temp due to path issues, move to target path now.
                    if record_temp_path and record_target_path and record_temp_path.exists():
                        try:
                            record_target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(record_temp_path), str(record_target_path))
                            print(f"[recording] saved -> {record_target_path}")
                        except Exception as exc:
                            print(f"[warn] move failed, kept temp file: {record_temp_path} ({exc})")
                    print("[recording] stop")
                    record_target_path = None
                    record_temp_path = None
    finally:
        if writer is not None:
            writer.release()
        if hands is not None:
            hands.close()
        if task_landmarker is not None:
            task_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
