"""
Plank Analyzer
--------------
Uses MediaPipe pose landmarks to:
  - Measure body line angle (shoulder → hip → ankle)
  - Detect form errors:
      * Hips too high (pike position)
      * Hips too low (sagging back)
      * Head not neutral (craning up or dropping)
      * Elbow/shoulder alignment check
  - Track continuous hold time in seconds
"""

import time
from utils.pose_utils import (
    calculate_angle, get_landmark_px, midpoint, visibility_ok
)

L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW    = 13
R_ELBOW    = 14
L_WRIST    = 15
R_WRIST    = 16
L_HIP      = 23
R_HIP      = 24
L_KNEE     = 25
R_KNEE     = 26
L_ANKLE    = 27
R_ANKLE    = 28
NOSE       = 0

BODY_ANGLE_MIN = 162   # degrees — below this = hips sagging
BODY_ANGLE_MAX = 198   # degrees — above this = hips piking


GRACE_PERIOD = 5.0   # seconds of bad form tolerated before hold resets

class PlankAnalyzer:
    def __init__(self):
        self.body_angle   = 180.0
        self.form_status  = "unknown"
        self.feedback     = []
        self.hold_start   = None
        self.hold_seconds = 0
        self._was_correct = False
        self._error_since = None   # when errors started (grace period tracking)

    # ------------------------------------------------------------------
    def analyze(self, landmarks, w, h):
        required = [L_SHOULDER, L_HIP, L_ANKLE, R_SHOULDER, R_HIP, R_ANKLE]
        if not visibility_ok(landmarks, required, threshold=0.4):
            self._was_correct = False
            self.hold_start   = None
            return self._state(["Stand/lie fully visible in frame"], "unknown")

        # ---- Landmarks ----
        l_shoulder = get_landmark_px(landmarks, L_SHOULDER, w, h)
        r_shoulder = get_landmark_px(landmarks, R_SHOULDER, w, h)
        l_hip      = get_landmark_px(landmarks, L_HIP,      w, h)
        r_hip      = get_landmark_px(landmarks, R_HIP,      w, h)
        l_ankle    = get_landmark_px(landmarks, L_ANKLE,    w, h)
        r_ankle    = get_landmark_px(landmarks, R_ANKLE,    w, h)
        nose       = get_landmark_px(landmarks, NOSE,       w, h)

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip      = midpoint(l_hip,      r_hip)
        mid_ankle    = midpoint(l_ankle,    r_ankle)

        # ---- Body line angle ----
        body_angle = calculate_angle(mid_shoulder, mid_hip, mid_ankle)
        self.body_angle = body_angle

        # ---- Form checks ----
        errors = []

        if body_angle < BODY_ANGLE_MIN:
            errors.append(f"Hips sagging ({body_angle:.0f}°) — engage your core & raise hips")
        elif body_angle > BODY_ANGLE_MAX:
            errors.append(f"Hips too high ({body_angle:.0f}°) — lower hips to form a straight line")

        # Head position: nose should be roughly in line with shoulders, not craned up
        nose_above_shoulder = mid_shoulder[1] - nose[1]   # positive = nose above shoulder
        if nose_above_shoulder > h * 0.12:
            errors.append("Head too high — look at the floor ahead of your hands")
        elif nose_above_shoulder < -h * 0.05:
            errors.append("Head dropping — keep neck neutral")

        # Hip symmetry (one side higher than other)
        hip_diff = abs(l_hip[1] - r_hip[1])
        if hip_diff > h * 0.06:
            errors.append("Hips are tilted — keep them level")

        is_correct = len(errors) == 0

        # ---- Hold time tracking (5s grace period) ----
        now = time.time()
        if is_correct:
            self._error_since = None          # clear any error grace timer
            if not self._was_correct:
                self.hold_start = now
            self.hold_seconds = now - (self.hold_start or now)
        else:
            if self._error_since is None:
                self._error_since = now       # start grace timer
            if now - self._error_since >= GRACE_PERIOD:
                # Bad form held for 5+ seconds — reset hold
                self.hold_start   = None
                self.hold_seconds = 0
            # else: within grace period, hold time keeps counting

        self._was_correct = is_correct

        status   = "correct" if is_correct else "error"
        feedback = errors if errors else [f"Perfect plank! Hold it — {self.hold_seconds:.0f}s"]
        self.form_status = status

        return self._state(feedback, status)

    # ------------------------------------------------------------------
    def _state(self, feedback, status="unknown"):
        return {
            "reps":         0,
            "hold_seconds": round(self.hold_seconds, 1),
            "body_angle":   round(self.body_angle, 1),
            "form_status":  status,
            "feedback":     feedback,
            "has_error":    status == "error",
        }

    def reset(self):
        self.body_angle   = 180.0
        self.form_status  = "unknown"
        self.feedback     = []
        self.hold_start   = None
        self.hold_seconds = 0
        self._was_correct = False
        self._error_since = None
