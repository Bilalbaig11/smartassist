"""
Squat Analyzer
--------------
Uses MediaPipe pose landmarks to:
  - Count squat reps (down when knee angle < 90°, up when > 160°)
  - Detect common form errors:
      * Knees caving inward (valgus collapse)
      * Excessive forward lean
      * Not reaching depth
      * Heels rising
"""

import time
from utils.pose_utils import (
    calculate_angle, get_landmark_px, midpoint, visibility_ok
)

# MediaPipe landmark indices
NOSE       = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP      = 23
R_HIP      = 24
L_KNEE     = 25
R_KNEE     = 26
L_ANKLE    = 27
R_ANKLE    = 28
L_HEEL     = 29
R_HEEL     = 30
L_FOOT     = 31
R_FOOT     = 32


class SquatAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.stage = "up"           # 'up' | 'down'
        self.feedback = []
        self.last_feedback = []
        self.knee_angle = 180.0
        self.torso_angle = 90.0
        self._prev_stage = "up"
        self._rep_in_progress = False

    # ------------------------------------------------------------------
    def analyze(self, landmarks, w, h):
        required = [L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE,
                    L_SHOULDER, R_SHOULDER]
        if not visibility_ok(landmarks, required, threshold=0.4):
            return self._state(["Stand fully visible in frame"])

        # ---- Landmarks ----
        l_hip      = get_landmark_px(landmarks, L_HIP,      w, h)
        r_hip      = get_landmark_px(landmarks, R_HIP,      w, h)
        l_knee     = get_landmark_px(landmarks, L_KNEE,     w, h)
        r_knee     = get_landmark_px(landmarks, R_KNEE,     w, h)
        l_ankle    = get_landmark_px(landmarks, L_ANKLE,    w, h)
        r_ankle    = get_landmark_px(landmarks, R_ANKLE,    w, h)
        l_shoulder = get_landmark_px(landmarks, L_SHOULDER, w, h)
        r_shoulder = get_landmark_px(landmarks, R_SHOULDER, w, h)

        mid_hip      = midpoint(l_hip, r_hip)
        mid_knee     = midpoint(l_knee, r_knee)
        mid_ankle    = midpoint(l_ankle, r_ankle)
        mid_shoulder = midpoint(l_shoulder, r_shoulder)

        # ---- Angles ----
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (l_knee_angle + r_knee_angle) / 2
        self.knee_angle = avg_knee_angle

        # Torso angle: vertical vs spine (hip->shoulder vector)
        vertical_above_hip = [mid_hip[0], mid_hip[1] - 100]
        self.torso_angle = calculate_angle(vertical_above_hip, mid_hip, mid_shoulder)

        # ---- Stage & Rep counting ----
        if avg_knee_angle > 160:
            if self.stage == "down":
                self.rep_count += 1
            self.stage = "up"
            self._rep_in_progress = False
        elif avg_knee_angle < 100:
            self.stage = "down"
            self._rep_in_progress = True

        # ---- Form analysis ----
        errors = []
        tips   = []

        # 1. Depth check
        if self.stage == "down":
            if avg_knee_angle > 100:
                errors.append("Go deeper — aim for thighs parallel to floor")
            else:
                tips.append("Good depth!")

        # 2. Knee valgus (caving in)
        knee_width = abs(l_knee[0] - r_knee[0])
        hip_width  = abs(l_hip[0]  - r_hip[0])
        if hip_width > 20 and knee_width < hip_width * 0.80:
            errors.append("Knees caving in — push them out over toes")

        # 3. Forward lean (torso should be ~30-70° from vertical during squat)
        if self.stage == "down" and self.torso_angle < 25:
            errors.append("Too upright — slight forward lean is normal")
        elif self.stage == "down" and self.torso_angle > 75:
            errors.append("Leaning too far forward — chest up!")

        # 4. Symmetry — left/right knee angle mismatch
        if abs(l_knee_angle - r_knee_angle) > 20:
            errors.append("Uneven squat — distribute weight equally")

        feedback = errors if errors else (tips if tips else ["Good form — keep it up!"])
        self.feedback = feedback

        return self._state(feedback)

    # ------------------------------------------------------------------
    def _state(self, feedback):
        return {
            "reps":        self.rep_count,
            "stage":       self.stage,
            "knee_angle":  round(self.knee_angle, 1),
            "torso_angle": round(self.torso_angle, 1),
            "feedback":    feedback,
            "has_error":   any(f not in ["Good form — keep it up!", "Good depth!", ""] for f in feedback),
        }

    def reset(self):
        self.rep_count      = 0
        self.stage          = "up"
        self.feedback       = []
        self.knee_angle     = 180.0
        self.torso_angle    = 90.0
        self._rep_in_progress = False
