"""
Session Logger
--------------
Creates one JSON log file per session in the /logs directory.
Also maintains a sessions_index.json for the logs listing page.
"""

import os
import json
import time
from datetime import datetime

LOGS_DIR = "logs"


def _ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


class SessionLogger:
    def __init__(self):
        self.session_id   = None
        self.filepath     = None
        self.data         = {}
        self._active      = False

    # ── Start a new session ───────────────────────────────────────
    def start(self, exercise: str, source: str, source_name: str = ""):
        _ensure_logs_dir()
        now = datetime.now()
        self.session_id = now.strftime("%Y%m%d_%H%M%S") + f"_{exercise}"
        self.filepath   = os.path.join(LOGS_DIR, f"session_{self.session_id}.json")

        self.data = {
            "session_id":    self.session_id,
            "exercise":      exercise,
            "source":        source,           # "webcam" | "video"
            "source_name":   source_name,
            "date":          now.strftime("%Y-%m-%d"),
            "start_time":    now.strftime("%H:%M:%S"),
            "end_time":      None,
            "duration_sec":  0,
            "reps":          0,
            "max_hold_sec":  0.0,
            "errors_total":  0,
            "corrections":   [],               # list of {time, message}
            "snapshots":     [],               # periodic stat snapshots
        }
        self._active   = True
        self._t_start  = time.time()
        self._last_snap = time.time()
        self._last_errors = []
        self._max_hold = 0.0
        self._save()
        return self.session_id

    # ── Call every frame with latest stats ───────────────────────
    def update(self, stats: dict):
        if not self._active:
            return

        now = time.time()

        # Reps
        self.data["reps"] = max(self.data["reps"], stats.get("reps", 0))

        # Hold time
        hold = stats.get("hold_seconds", 0.0)
        if hold > self._max_hold:
            self._max_hold = hold
            self.data["max_hold_sec"] = round(hold, 1)

        # Error / correction logging (deduplicated)
        feedback  = stats.get("feedback", [])
        has_error = stats.get("has_error", False)
        if has_error and feedback:
            msg = feedback[0]
            if msg not in self._last_errors:
                self._last_errors = [msg]
                elapsed = round(now - self._t_start, 1)
                self.data["corrections"].append({
                    "time_sec": elapsed,
                    "message":  msg,
                })
                self.data["errors_total"] += 1
        else:
            self._last_errors = []

        # Periodic snapshot every 5 seconds
        if now - self._last_snap >= 5:
            self._last_snap = now
            snap = {
                "t":     round(now - self._t_start, 1),
                "reps":  stats.get("reps", 0),
                "hold":  round(hold, 1),
            }
            if "knee_angle" in stats:
                snap["knee_angle"] = stats["knee_angle"]
            if "body_angle" in stats:
                snap["body_angle"] = stats["body_angle"]
            self.data["snapshots"].append(snap)

        self.data["duration_sec"] = round(now - self._t_start, 1)
        self._save()

    # ── End the session ───────────────────────────────────────────
    def end(self):
        if not self._active:
            return
        self._active = False
        now = datetime.now()
        self.data["end_time"]     = now.strftime("%H:%M:%S")
        self.data["duration_sec"] = round(time.time() - self._t_start, 1)
        self._save()
        self._update_index()
        return self.data

    # ── Helpers ───────────────────────────────────────────────────
    def _save(self):
        if self.filepath:
            with open(self.filepath, "w") as f:
                json.dump(self.data, f, indent=2)

    def _update_index(self):
        index_path = os.path.join(LOGS_DIR, "sessions_index.json")
        index = []
        if os.path.exists(index_path):
            try:
                with open(index_path) as f:
                    index = json.load(f)
            except Exception:
                index = []

        # Summary entry
        summary = {
            "session_id":   self.data["session_id"],
            "date":         self.data["date"],
            "start_time":   self.data["start_time"],
            "end_time":     self.data["end_time"],
            "duration_sec": self.data["duration_sec"],
            "exercise":     self.data["exercise"],
            "source":       self.data["source"],
            "source_name":  self.data["source_name"],
            "reps":         self.data["reps"],
            "max_hold_sec": self.data["max_hold_sec"],
            "errors_total": self.data["errors_total"],
        }
        # Remove old entry for same session_id if exists, then prepend
        index = [s for s in index if s.get("session_id") != self.session_id]
        index.insert(0, summary)

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)


# ── Module-level helpers ──────────────────────────────────────────────

def load_index():
    _ensure_logs_dir()
    index_path = os.path.join(LOGS_DIR, "sessions_index.json")
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path) as f:
            return json.load(f)
    except Exception:
        return []


def load_session(session_id: str):
    _ensure_logs_dir()
    path = os.path.join(LOGS_DIR, f"session_{session_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def delete_session(session_id: str):
    _ensure_logs_dir()
    path = os.path.join(LOGS_DIR, f"session_{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
    index_path = os.path.join(LOGS_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
            index = [s for s in index if s.get("session_id") != session_id]
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass
