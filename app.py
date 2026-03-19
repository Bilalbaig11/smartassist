"""
SmartAssist — Flask Backend  (v4)
==================================
Pages:  /  |  /session  |  /logs

Key API endpoints:
  POST /start_session      {exercise, source, source_name?, groq_api_key?}
  POST /stop_session
  POST /set_exercise       {exercise}
  POST /reset
  POST /upload_video       (multipart)
  POST /session/goal       {exercise, target_reps?, target_seconds?}
  GET  /session/goal       → current goal
  POST /voice/chat         (multipart audio)  → {transcript, reply}
  POST /voice/auto_coach   {trigger, details?} → {reply} | {skip:true}
  POST /voice/config       {groq_api_key}
  GET  /stats
  GET  /api/logs  |  GET /api/logs/<id>  |  DELETE /api/logs/<id>
  GET  /health
"""

import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Load .env if present — GROQ_API_KEY will be read from environment
from dotenv import load_dotenv
load_dotenv()

from exercises.squat import SquatAnalyzer
from exercises.plank import PlankAnalyzer
from utils.session_logger import SessionLogger, load_index, load_session, delete_session
from utils.voice_coach import VoiceCoach

# ── App ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ── Pose model ────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_lite.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe pose model (~5 MB)…")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task",
        MODEL_PATH,
    )
    print("Model ready.")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),
]

# ── Global state ──────────────────────────────────────────────────────
_lock            = threading.Lock()
current_exercise = "squat"
video_source     = "webcam"
video_filename   = ""
analyzers        = {"squat": SquatAnalyzer(), "plank": PlankAnalyzer()}
latest_stats: dict  = {}
session_active   = False
session_logger   = SessionLogger()
_stop_stream     = threading.Event()

# Voice coach — try env key first, else wait for UI
_env_key = os.getenv("GROQ_API_KEY", "").strip()
voice_coach: VoiceCoach | None = VoiceCoach(_env_key) if _env_key else None
if _env_key:
    print("✓ Groq API key loaded from environment.")
else:
    print("⚠  No GROQ_API_KEY in environment — enter it in the Setup UI.")

# ── Exercise goal (in-memory) ─────────────────────────────────────────
# {
#   "exercise":       "squat" | "plank",
#   "target_reps":    int | None,
#   "target_seconds": float | None,
# }
current_goal: dict = {}


def _get_or_create_coach(api_key: str | None) -> VoiceCoach | None:
    global voice_coach
    if api_key:
        voice_coach = VoiceCoach(api_key=api_key)
    return voice_coach


# ─────────────────────────────────────────────────────────────────────
#  Drawing helpers (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────

def _put_text_with_bg(frame, text, pos, font_scale=0.65, color=(255,255,255),
                      bg_color=(0,0,0), thickness=2, bg_alpha=0.60):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(pos[0]), int(pos[1])
    pad  = 6
    ov   = frame.copy()
    cv2.rectangle(ov, (x-pad, y-th-pad), (x+tw+pad, y+baseline+pad), bg_color, -1)
    cv2.addWeighted(ov, bg_alpha, frame, 1-bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_skeleton(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (80, 200, 255), 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 180), -1, cv2.LINE_AA)


def draw_squat_overlay(frame, stats, h, w):
    reps, stage, angle = stats.get("reps",0), stats.get("stage","up").upper(), stats.get("knee_angle",0)
    _put_text_with_bg(frame, "SQUAT",          (12,32),  0.85, (0,255,200),   (20,20,20))
    _put_text_with_bg(frame, f"REPS  {reps}",  (12,72),  0.75, (255,255,255), (30,30,30))
    _put_text_with_bg(frame, f"STAGE  {stage}",(12,108), 0.65, (200,200,200), (30,30,30))
    _put_text_with_bg(frame, f"KNEE  {angle}", (12,140), 0.65, (200,200,200), (30,30,30))
    sc = (0,220,80) if stage=="UP" else (0,140,255)
    _put_text_with_bg(frame, stage, (w-110,38), 0.85, sc, (10,10,10))


def draw_plank_overlay(frame, stats, h, w):
    angle, hold, status = stats.get("body_angle",0), stats.get("hold_seconds",0), stats.get("form_status","unknown")
    sc, label = ((0,220,80),"CORRECT") if status=="correct" else ((30,80,255),"FIX FORM")
    _put_text_with_bg(frame, "PLANK",              (12,32),  0.85, (0,255,200),   (20,20,20))
    _put_text_with_bg(frame, f"HOLD  {hold:.0f}s", (12,72),  0.75, (255,255,255), (30,30,30))
    _put_text_with_bg(frame, f"BODY  {angle}",     (12,108), 0.65, (200,200,200), (30,30,30))
    _put_text_with_bg(frame, label, (w-160,38), 0.85, sc, (10,10,10))
    bx, bt, bb = w-30, 80, h-80
    bh = bb - bt
    ry = int(bt + bh*(1-np.clip((angle-140)/80,0,1)))
    bc = (0,220,80) if (162<=angle<=198) else (30,80,255)
    cv2.rectangle(frame,(bx-12,bt),(bx,bb),(40,40,40),-1)
    cv2.rectangle(frame,(bx-12,ry),(bx,bb),bc,-1)
    cv2.line(frame,(bx-18,int(bt+bh*0.5)),(bx+4,int(bt+bh*0.5)),(255,255,255),1)


def draw_feedback(frame, feedback, h, w, has_error):
    n, start = len(feedback), h - 36*len(feedback) - 12
    for i, msg in enumerate(feedback):
        col = (30,100,255) if has_error and i==0 else (0,220,80)
        bg  = (10,10,30)   if has_error and i==0 else (10,30,10)
        _put_text_with_bg(frame, msg, (12, start+i*36+22), 0.62, col, bg, thickness=1, bg_alpha=0.6)


# ─────────────────────────────────────────────────────────────────────
#  Frame generator
# ─────────────────────────────────────────────────────────────────────

def _make_landmarker():
    opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(opts)


def generate_frames():
    global latest_stats
    with _lock:
        src, vfn = video_source, video_filename

    if src == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        is_video, delay = False, 0
    else:
        cap = cv2.VideoCapture(os.path.join(UPLOAD_FOLDER, vfn))
        is_video = True
        delay    = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30)

    _stop_stream.clear()

    with _make_landmarker() as lm_model:
        while not _stop_stream.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok: break
                else:
                    time.sleep(0.05); continue

            h, w = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = lm_model.detect(mp_image)

            with _lock:
                ex, analyzer, s_active = current_exercise, analyzers[current_exercise], session_active

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                _draw_skeleton(frame, lms, w, h)
                stats = analyzer.analyze(lms, w, h)
                with _lock:
                    latest_stats = stats
                    if s_active: session_logger.update(stats)
                (draw_squat_overlay if ex=="squat" else draw_plank_overlay)(frame, stats, h, w)
                draw_feedback(frame, stats.get("feedback",[]), h, w, stats.get("has_error",False))
            else:
                _put_text_with_bg(frame,"No person detected — step into frame",
                                  (int(w*.08),int(h*.5)),0.7,(200,80,40),(10,10,10))
                with _lock: latest_stats = {}

            lbl = "WEBCAM" if src=="webcam" else os.path.basename(vfn)[:22]
            _put_text_with_bg(frame, lbl, (w-190,h-16), 0.45, (90,90,90), (0,0,0), thickness=1)

            if is_video: time.sleep(delay)

            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok2:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    cap.release()


# ─────────────────────────────────────────────────────────────────────
#  Page routes
# ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    # Pass whether voice is pre-configured from env
    return render_template("home.html", voice_from_env=bool(voice_coach))


@app.route("/session")
def session_page():
    return render_template("session.html",
                           exercise=current_exercise,
                           source=video_source,
                           source_name=video_filename,
                           goal=current_goal)


@app.route("/logs")
def logs_page():
    return render_template("logs.html")


# ─────────────────────────────────────────────────────────────────────
#  Core session API
# ─────────────────────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    with _lock:
        data = dict(latest_stats)
        data.update(exercise=current_exercise, session_active=session_active,
                    source=video_source)
    return jsonify(data)


@app.route("/start_session", methods=["POST"])
def start_session():
    global session_active, current_exercise, video_source, video_filename, current_goal
    body    = request.get_json(silent=True) or {}
    ex      = body.get("exercise",    "squat").lower()
    src     = body.get("source",      "webcam").lower()
    srn     = body.get("source_name", "")
    api_key = body.get("groq_api_key","").strip()

    # Goal from request (optional — can also be set via /session/goal)
    goal_body = body.get("goal", {})
    with _lock:
        current_exercise = ex
        video_source, video_filename = src, srn
        analyzers[ex].reset()
        session_active = True
        if goal_body:
            current_goal = {
                "exercise":       ex,
                "target_reps":    int(goal_body["target_reps"])    if goal_body.get("target_reps")    else None,
                "target_seconds": float(goal_body["target_seconds"]) if goal_body.get("target_seconds") else None,
            }

    coach = _get_or_create_coach(api_key or None)
    if coach: coach.clear_history()

    sid = session_logger.start(ex, src, srn)
    _stop_stream.clear()
    return jsonify({"status":"ok","session_id":sid,"exercise":ex,
                    "source":src,"voice_ready":voice_coach is not None})


@app.route("/stop_session", methods=["POST"])
def stop_session():
    global session_active
    with _lock: session_active = False
    _stop_stream.set()
    return jsonify({"status":"ok","session":session_logger.end()})


@app.route("/set_exercise", methods=["POST"])
def set_exercise():
    global current_exercise
    body = request.get_json(silent=True) or {}
    ex   = body.get("exercise","squat").lower()
    if ex not in analyzers:
        return jsonify({"error":"Unknown exercise"}), 400
    with _lock:
        current_exercise = ex
        analyzers[ex].reset()
    return jsonify({"status":"ok","exercise":ex})


@app.route("/reset", methods=["POST"])
def reset():
    with _lock: analyzers[current_exercise].reset()
    return jsonify({"status":"ok"})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error":"No file"}), 400
    fname = secure_filename(f.filename)
    f.save(os.path.join(UPLOAD_FOLDER, fname))
    return jsonify({"status":"ok","filename":fname})


# ─────────────────────────────────────────────────────────────────────
#  Goal API
# ─────────────────────────────────────────────────────────────────────

@app.route("/session/goal", methods=["GET"])
def get_goal():
    with _lock:
        return jsonify(current_goal)


@app.route("/session/goal", methods=["POST"])
def set_goal():
    global current_goal
    body = request.get_json(silent=True) or {}
    ex   = body.get("exercise", current_exercise).lower()
    g = {
        "exercise":       ex,
        "target_reps":    int(body["target_reps"])      if body.get("target_reps")    else None,
        "target_seconds": float(body["target_seconds"])  if body.get("target_seconds") else None,
    }
    with _lock: current_goal = g
    # Reset goal-achieved flag in coach so it can fire again for a new goal
    if voice_coach: voice_coach._goal_achieved_fired = False
    return jsonify({"status":"ok","goal":g})


# ─────────────────────────────────────────────────────────────────────
#  Voice API
# ─────────────────────────────────────────────────────────────────────

@app.route("/voice/config", methods=["POST"])
def voice_config():
    body    = request.get_json(silent=True) or {}
    api_key = body.get("groq_api_key","").strip()
    if not api_key:
        return jsonify({"error":"groq_api_key required"}), 400
    _get_or_create_coach(api_key)
    return jsonify({"status":"ok","voice_ready":True})


@app.route("/voice/chat", methods=["POST"])
def voice_chat():
    """PTT — user recorded audio → transcribe → LLM reply."""
    coach = voice_coach
    if not coach:
        return jsonify({"error":"Voice not configured — add Groq API key"}), 503

    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error":"No audio"}), 400
    audio_bytes = audio_file.read()
    if not audio_bytes:
        return jsonify({"error":"Empty audio"}), 400

    fname = audio_file.filename or "audio.webm"

    try:
        transcript = coach.transcribe(audio_bytes, filename=fname)
    except Exception as e:
        return jsonify({"error":f"Transcription failed: {e}"}), 500

    if not transcript:
        return jsonify({"error":"Could not understand audio"}), 422

    with _lock:
        ex    = request.form.get("exercise", current_exercise)
        stats = dict(latest_stats)

    try:
        reply = coach.get_response(transcript, stats, ex)
    except Exception as e:
        return jsonify({"error":f"LLM failed: {e}"}), 500

    return jsonify({"status":"ok","transcript":transcript,"reply":reply,"exercise":ex})


@app.route("/voice/auto_coach", methods=["POST"])
def voice_auto_coach():
    """
    System-triggered coaching (no user button press).
    Body: { "trigger": "form_error"|"goal_achieved"|"struggling"|"encouragement",
            "details": {...}  (optional extra info) }
    Returns: { "reply": "..." } | { "skip": true }
    """
    coach = voice_coach
    if not coach:
        return jsonify({"skip": True})

    body    = request.get_json(silent=True) or {}
    trigger = body.get("trigger", "form_error")
    details = body.get("details")

    with _lock:
        ex    = current_exercise
        stats = dict(latest_stats)

    try:
        reply, throttled = coach.auto_coach(trigger, stats, ex, details)
    except Exception as e:
        return jsonify({"skip": True, "error": str(e)})

    if throttled or not reply:
        return jsonify({"skip": True})

    return jsonify({"reply": reply})


# ─────────────────────────────────────────────────────────────────────
#  Log API
# ─────────────────────────────────────────────────────────────────────

@app.route("/api/logs")
def api_logs():
    return jsonify(load_index())


@app.route("/api/logs/<sid>")
def api_log_detail(sid):
    d = load_session(sid)
    return jsonify(d) if d else (jsonify({"error":"Not found"}), 404)


@app.route("/api/logs/<sid>", methods=["DELETE"])
def api_log_delete(sid):
    delete_session(sid)
    return jsonify({"status":"ok"})


@app.route("/health")
def health():
    return jsonify({
        "status":      "running",
        "exercise":    current_exercise,
        "source":      video_source,
        "session":     session_active,
        "voice_ready": voice_coach is not None,
        "voice_from_env": bool(_env_key),
        "goal":        current_goal,
    })


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*54)
    print("  SmartAssist — AI Exercise Coach")
    print("  by Bilal Baig")
    print("  Open http://localhost:5000 in your browser")
    print("="*54 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
