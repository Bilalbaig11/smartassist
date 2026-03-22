"""
pose_engine.py
--------------
MediaPipe wrapper — mirrors generate_frames() in app.py exactly:
  same model, same confidence thresholds, same image conversion,
  same landmark access pattern.
"""

import os, base64, uuid, urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

# ── same as app.py ────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_lite.task"

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (~5 MB)…")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            MODEL_PATH,
        )
        print("Model ready.")


def make_landmarker():
    """Exact copy of _make_landmarker() from app.py — same thresholds."""
    ensure_model()
    opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(opts)


def draw_skeleton(frame, landmarks, w, h, color=(80, 200, 255)):
    """Exact copy of _draw_skeleton() from app.py."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 180), -1, cv2.LINE_AA)


def extract_video_frames(path, interval=0.2):
    """
    Extract one frame every `interval` seconds.
    Returns (frames, fps, duration_sec).

    Robust fps detection — browser-recorded webm/mp4 often reports 0 fps.
    Falls back to 30 if the reported value is out of a sane range.
    """
    cap   = cv2.VideoCapture(str(path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    if not (5 <= fps <= 120):
        fps = 30.0

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration     = total_frames / fps if total_frames > 0 else 0.0
    step         = max(1, int(fps * interval))

    raw = []; idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            raw.append(frame)
        idx += 1
    cap.release()
    return raw, fps, duration


def _frame_to_b64(frame, max_dim=380):
    h, w = frame.shape[:2]
    s = min(max_dim / max(w, h), 1.0)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


def process_frames(
    raw_frames: list,
    feature_fn,
    progress_cb=None,          # optional callback(current, total, kept, skipped)
) -> tuple[list, int]:
    """
    Run MediaPipe on every frame using the same pipeline as app.py.
    AUTO-DISCARDS frames where no pose is detected or features are None.
    Calls progress_cb(i, total, kept, skipped) after every frame if supplied.
    Returns (detected_frames, total_skipped).
    """
    out     = []
    skipped = 0
    total   = len(raw_frames)

    with make_landmarker() as lm_model:
        for i, frame in enumerate(raw_frames):
            h, w = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lm_model.detect(mp_img)

            if not result.pose_landmarks:
                skipped += 1
            else:
                lms      = result.pose_landmarks[0]
                features = feature_fn(lms, w, h)
                if features is None:
                    skipped += 1
                else:
                    ann = frame.copy()
                    draw_skeleton(ann, lms, w, h, (80, 200, 255))
                    out.append({
                        "id":       str(uuid.uuid4())[:8],
                        "img_b64":  _frame_to_b64(ann),
                        "features": features,
                        "detected": True,
                        "label":    None,
                    })

            if progress_cb:
                progress_cb(i + 1, total, len(out), skipped)

    return out, skipped