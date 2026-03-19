import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b, given three 2D points a, b, c.
    Returns angle in degrees (0-180).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def get_landmark_px(landmarks, idx, w, h):
    """Return landmark as [x_px, y_px]."""
    lm = landmarks[idx]
    return [lm.x * w, lm.y * h]


def get_landmark_norm(landmarks, idx):
    """Return landmark as [x_norm, y_norm] (0-1 range)."""
    lm = landmarks[idx]
    return [lm.x, lm.y]


def midpoint(a, b):
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


def visibility_ok(landmarks, indices, threshold=0.5):
    """Check that all required landmarks are visible enough."""
    return all(landmarks[i].visibility > threshold for i in indices)
