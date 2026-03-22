"""
features.py
-----------
Feature extraction for squat and plank calibration.

Helpers are direct copies of utils/pose_utils.py so the numbers are
identical to what the live analyzers compute.

Squat classes:  correct | knees_inward | knees_outward
Plank classes:  correct | hips_too_low | hips_too_high
"""

import numpy as np

# ── Landmark indices (same as exercises/squat.py & plank.py) ─────────
NOSE       = 0
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28

# ── Helpers — exact copies of utils/pose_utils.py ────────────────────

def calculate_angle(a, b, c):
    """Angle at point b given three 2-D points a, b, c. Returns degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def get_landmark_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def midpoint(a, b):
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def visibility_ok(landmarks, indices, threshold=0.4):
    """Same threshold as SquatAnalyzer / PlankAnalyzer."""
    return all(landmarks[i].visibility > threshold for i in indices)


# ── Label definitions ─────────────────────────────────────────────────

SQUAT_LABELS = [
    {"id": "correct",       "label": "Correct Form",    "color": "#00e5a0"},
    {"id": "knees_inward",  "label": "Knees Caving In", "color": "#ff4646"},
    {"id": "knees_outward", "label": "Knees Too Wide",  "color": "#3ba7ff"},
]

PLANK_LABELS = [
    {"id": "correct",       "label": "Correct Form",  "color": "#00e5a0"},
    {"id": "hips_too_low",  "label": "Hips Too Low",  "color": "#ff4646"},
    {"id": "hips_too_high", "label": "Hips Too High", "color": "#9d6cff"},
]

# Feature columns used for SVM training
SQUAT_FEAT_COLS = ["avg_knee_angle", "knee_hip_ratio", "l_knee_angle", "r_knee_angle"]
PLANK_FEAT_COLS = ["body_angle"]

# ── Original hardcoded thresholds from exercises/ ────────────────────
# Source: exercises/squat.py  line 101   knee_width < hip_width * 0.80
#         exercises/plank.py  line 33-34  BODY_ANGLE_MIN=162, BODY_ANGLE_MAX=198
ORIGINAL_THRESHOLDS = {
    "squat": {
        "knees_inward": {
            "feature":   "knee_hip_ratio",
            "direction": "<",
            "threshold": 0.80,
            "note":      "Original: knee_width < hip_width * 0.80  (squat.py line 101)",
        },
        # knees_outward: NOT present in original code at all
    },
    "plank": {
        "hips_too_low": {
            "feature":   "body_angle",
            "direction": "<",
            "threshold": 162.0,
            "note":      "Original: BODY_ANGLE_MIN = 162  (plank.py line 33)",
        },
        "hips_too_high": {
            "feature":   "body_angle",
            "direction": ">",
            "threshold": 198.0,
            "note":      "Original: BODY_ANGLE_MAX = 198  (plank.py line 34)",
        },
    },
}


# ── Feature extractors ────────────────────────────────────────────────

def extract_squat_features(landmarks, w, h):
    """
    Returns dict with:
      avg_knee_angle  — average of L+R knee bend angle (hip→knee→ankle)
      knee_hip_ratio  — knee_width / hip_width  (< 1 = caving, > 1.2 = too wide)
      l_knee_angle    — left knee angle
      r_knee_angle    — right knee angle
    """
    required = [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK, L_SHO, R_SHO]
    if not visibility_ok(landmarks, required, threshold=0.4):
        return None

    lh = get_landmark_px(landmarks, L_HIP, w, h)
    rh = get_landmark_px(landmarks, R_HIP, w, h)
    lk = get_landmark_px(landmarks, L_KNE, w, h)
    rk = get_landmark_px(landmarks, R_KNE, w, h)
    la = get_landmark_px(landmarks, L_ANK, w, h)
    ra = get_landmark_px(landmarks, R_ANK, w, h)

    lka = calculate_angle(lh, lk, la)
    rka = calculate_angle(rh, rk, ra)
    avg = (lka + rka) / 2.0

    knee_width = abs(lk[0] - rk[0])
    hip_width  = abs(lh[0] - rh[0])
    khr        = knee_width / (hip_width + 1e-8)

    return {
        "avg_knee_angle": round(avg,  2),
        "l_knee_angle":   round(lka,  2),
        "r_knee_angle":   round(rka,  2),
        "knee_hip_ratio": round(khr,  4),
    }


def extract_plank_features(landmarks, w, h):
    """
    Returns dict with:
      body_angle — angle at hip midpoint (shoulder→hip→ankle)
                   perfect plank ≈ 180°, sagging < 162°, piking > 198°
    """
    required = [L_SHO, R_SHO, L_HIP, R_HIP, L_ANK, R_ANK]
    if not visibility_ok(landmarks, required, threshold=0.4):
        return None

    ls = get_landmark_px(landmarks, L_SHO, w, h)
    rs = get_landmark_px(landmarks, R_SHO, w, h)
    lh = get_landmark_px(landmarks, L_HIP, w, h)
    rh = get_landmark_px(landmarks, R_HIP, w, h)
    la = get_landmark_px(landmarks, L_ANK, w, h)
    ra = get_landmark_px(landmarks, R_ANK, w, h)

    ms = midpoint(ls, rs)
    mh = midpoint(lh, rh)
    ma = midpoint(la, ra)

    body_angle = calculate_angle(ms, mh, ma)

    return {"body_angle": round(body_angle, 2)}