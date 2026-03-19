<div align="center">

# 🏋️ SmartAssist — AI Exercise Coach

**Real-time pose analysis + AI voice coaching, right in your browser.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-brightgreen?logo=google)](https://mediapipe.dev)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-orange)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

SmartAssist watches your form in real time, counts your reps, and gives you spoken coaching — like having a personal trainer watching every move.

*Built by [Bilal Baig](https://github.com/bilalbaig) · Final Year Project*

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🦵 **Squat Coach** | Automatic rep counting with depth, knee valgus, and posture checks |
| 💪 **Plank Coach** | Hold timer that only counts during correct form — alignment feedback in real time |
| 🎙️ **AI Voice Coach** | Hold the mic and ask anything mid-workout — powered by Groq Whisper + LLaMA 3.3 70B |
| 🔔 **Auto Coaching** | SmartAssist speaks unprompted on form errors, goal hits, and when you need a push |
| 🎯 **Goal Tracking** | Set a rep or hold-time target — live progress bar tracks it during your session |
| 📊 **Workout History** | Every session saved: duration, reps, best hold, and timestamped form corrections |
| 🎬 **Video Upload** | Analyse your workout from an MP4 instead of a live webcam |

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10 or 3.11** — [python.org](https://python.org) (check "Add to PATH" on Windows)
- A **webcam** (or an MP4 video file)
- A free **[Groq API key](https://console.groq.com)** *(optional — only needed for voice coaching)*

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Bilalbaig11/smartassist.git
cd smartassist

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key (optional)
cp .env.example .env
# Open .env and paste your Groq key

# 5. Run
python app.py
```

Open **http://localhost:5000** in your browser. The MediaPipe pose model (~5 MB) downloads automatically on first run.

---

## 🎙️ Voice Coaching Setup

Voice coaching uses Groq's ultra-fast inference — get a free key at [console.groq.com](https://console.groq.com).

**Option A — `.env` file (recommended):**
```env
GROQ_API_KEY=gsk_your_key_here
```

**Option B — enter in the UI:**  
Paste your key into the *Voice Coach* field on the Setup page. No restart needed.

Without a key, all visual coaching (rep counting, form feedback, history) still works — only spoken audio is disabled.

---

## 🗂️ Project Structure

```
smartassist/
├── app.py                    ← Flask server · video streaming · all API routes
├── plank_calibrate.py        ← Optional: calibrate plank thresholds to your camera
├── requirements.txt
├── .env.example              ← Copy to .env and add your Groq key
│
├── exercises/
│   ├── squat.py              ← Rep counting + form analysis
│   └── plank.py              ← Hold timer + alignment checks
│
├── utils/
│   ├── pose_utils.py         ← Angle calculations, landmark helpers
│   ├── session_logger.py     ← Per-session JSON logging
│   └── voice_coach.py        ← Groq transcription + LLM coaching responses
│
├── templates/
│   ├── home.html             ← Setup page (exercise, source, goal, API key)
│   ├── session.html          ← Live workout page
│   └── logs.html             ← Workout history with detail drawer
│
├── static/
│   └── base.css              ← Shared design tokens
│
├── uploads/                  ← Uploaded video files (git-ignored)
└── logs/                     ← Saved session JSON files (git-ignored)
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Pose Detection | [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |
| Web Server | [Flask 3](https://flask.palletsprojects.com) |
| Video Streaming | OpenCV · MJPEG over HTTP |
| AI Transcription | Groq · `whisper-large-v3-turbo` |
| AI Coaching | Groq · `llama-3.3-70b-versatile` |
| Text-to-Speech | Browser Web Speech API |
| Frontend | Vanilla JS · CSS custom properties |

---

## 📐 How It Works

```
Webcam / Video
      │
      ▼
  OpenCV Frame
      │
      ▼
MediaPipe Pose Landmarker
  (33 body landmarks)
      │
      ├──► SquatAnalyzer  →  reps, stage, knee/torso angles, form errors
      │
      └──► PlankAnalyzer  →  hold timer, body-line angle, head/hip checks
                │
                ▼
         Flask API (/stats)
                │
         Browser polls @ 240ms
                │
         ┌──────┴──────┐
         │             │
      UI Update    Auto-Coach
    (stats, gauges)  trigger?
                         │
                    Groq LLaMA 3.3
                    (≤25 word reply)
                         │
                   Web Speech API
                   (spoken aloud)
```

---

## 🎛️ Plank Calibration *(optional)*

The default plank thresholds work for most setups. If your camera angle is unusual, run the calibration tool for personalised values:

```bash
python plank_calibrate.py
```

> **Camera placement matters:** position your camera to the **side** of your body at waist height for best results. Front-facing cameras produce unreliable body-angle readings.

Follow the on-screen prompts to record three poses (correct plank, hips sagging, hips piked). The tool outputs values you can paste directly into `exercises/plank.py`.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/start_session` | Begin a session `{exercise, source, groq_api_key?, goal?}` |
| `POST` | `/stop_session` | End session, save log |
| `GET` | `/stats` | Current frame stats (reps, angles, feedback) |
| `GET` | `/video_feed` | MJPEG stream |
| `POST` | `/set_exercise` | Switch exercise mid-session |
| `POST` | `/reset` | Reset rep / hold counters |
| `POST` | `/upload_video` | Upload an MP4 for analysis |
| `POST` | `/session/goal` | Set or update the active goal |
| `POST` | `/voice/chat` | PTT — send audio, get transcript + reply |
| `POST` | `/voice/auto_coach` | System-triggered coaching event |
| `GET` | `/api/logs` | All session summaries |
| `GET` | `/api/logs/<id>` | Full session detail |
| `DELETE` | `/api/logs/<id>` | Delete a session |

---

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| Camera not working | Ensure no other app is using the camera; try unplugging/replugging |
| Port 5000 in use | Change `port=5000` → `port=5001` at the bottom of `app.py` |
| Voice not working | Check your API key starts with `gsk_` and is pasted correctly |
| No person detected | Ensure good lighting and that your full body (head to feet) is visible |
| Slow on old hardware | Switch to `pose_landmarker_lite` (already default) or reduce resolution in `app.py` |
| `cv2` install fails | Use `pip install opencv-python-headless` instead on servers |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<sub>SmartAssist · Final Year Project · Built by Bilal Baig</sub>
</div>