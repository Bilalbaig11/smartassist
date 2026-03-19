"""
VoiceCoach  (SmartAssist v4)
-----------------------------
Handles:
  1. Transcription  — Groq Whisper
  2. PTT coaching   — user speaks, LLM responds
  3. Auto-coaching  — system-triggered on form errors, goal events, struggling
     with per-trigger throttle so it never overwhelms the user

Trigger types  (used by /voice/auto_coach):
  "form_error"     — bad form detected              → throttle 6 s
  "goal_achieved"  — rep/hold target reached         → one-shot (fire once)
  "struggling"     — user pausing/dropped plank      → throttle 20 s
  "encouragement"  — gentle mid-session nudge        → throttle 30 s
"""

import json
import time
from groq import Groq


# ─────────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────────

PTT_SYSTEM = """You are SmartAssist, an elite personal fitness coach in a real-time AI exercise app.

PERSONALITY: Warm, direct, motivating — like a knowledgeable gym buddy, not a robot.

PTT RESPONSE RULES:
- MAX 2 short sentences. User is mid-workout.
- Use the exact numbers from sensor data (reps, angles, hold time).
- Casual spoken tone — contractions, no jargon.
- Never say "Based on data" — just coach naturally.
- No bullet points, headers, or lists — pure speech.
- Under 30 words ideally.

SENSOR CONTEXT comes as JSON before the user's question.
"""

AUTO_SYSTEM = """You are SmartAssist, an elite personal fitness coach. You speak automatically when something important happens during a workout — the user did NOT press a button. You are speaking INTO THE ROOM, unprompted.

RULES FOR AUTO-COACHING:
- MAX 1-2 very short sentences. Sound natural, not like a notification.
- Sound like you're physically present in the room.
- For form errors: be specific about the mistake, give one clear fix.
- For goal achieved: celebrate with genuine energy.
- For struggling: be warm and encouraging, not pushy.
- Never say "I notice" or "It appears" — just speak directly.
- Under 25 words. Tight and punchy.

TRIGGER TYPES you will receive:
  form_error    — user has a specific form issue right now
  goal_achieved — user just hit their rep/hold target
  struggling    — user paused reps or dropped their plank early
  encouragement — mid-workout check-in

EXAMPLES:
form_error (knees caving): "Push your knees out! Imagine spreading the floor with your feet."
goal_achieved (10 squats): "Yes! Ten squats done — that's your goal crushed!"
struggling (squat pause):  "Come on, you've got more in you — one more rep!"
struggling (plank drop):   "That's okay, shake it out and get back in position."
encouragement:             "Looking strong — keep that core tight!"
"""


# ─────────────────────────────────────────────────────────────────────
#  Throttle config  (seconds between auto-speaks per trigger type)
# ─────────────────────────────────────────────────────────────────────
THROTTLE = {
    "form_error":    6.0,
    "struggling":   20.0,
    "encouragement":30.0,
    "goal_achieved":  0,    # one-shot — handled separately
}


class VoiceCoach:
    def __init__(self, api_key: str):
        self.client  = Groq(api_key=api_key)
        self._history: list[dict] = []
        self._max_history_turns   = 3

        # Throttle state: {trigger_type: last_fire_timestamp}
        self._last_fire: dict[str, float] = {}
        self._goal_achieved_fired = False

    # ── PTT Transcription ─────────────────────────────────────────────
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> str:
        result = self.client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",
        )
        return (result or "").strip()

    # ── PTT Coaching (user-triggered) ─────────────────────────────────
    def get_response(self, user_message: str, stats: dict, exercise: str) -> str:
        context  = self._build_context(stats, exercise)
        messages = [{"role": "system", "content": PTT_SYSTEM}]
        messages += self._history
        messages.append({
            "role": "user",
            "content": (
                f"[LIVE SENSOR DATA]\n{json.dumps(context, indent=2)}\n\n"
                f"[USER SAID]\n{user_message}"
            ),
        })

        resp  = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.72,
            max_tokens=90,
            top_p=0.9,
        )
        reply = resp.choices[0].message.content.strip()

        # Rolling history — clean (no JSON noise)
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": reply})
        max_msg = self._max_history_turns * 2
        if len(self._history) > max_msg:
            self._history = self._history[-max_msg:]

        return reply

    # ── Auto-Coach (system-triggered) ─────────────────────────────────
    def auto_coach(
        self,
        trigger:  str,
        stats:    dict,
        exercise: str,
        details:  dict | None = None,
    ) -> tuple[str | None, bool]:
        """
        Returns (reply, was_throttled).
        reply=None and was_throttled=True  → caller should skip TTS.
        """
        now = time.time()

        # One-shot goal achieved — fire only once per session
        if trigger == "goal_achieved":
            if self._goal_achieved_fired:
                return None, True
            self._goal_achieved_fired = True
        else:
            # Throttle check
            throttle_secs = THROTTLE.get(trigger, 10.0)
            last = self._last_fire.get(trigger, 0.0)
            if now - last < throttle_secs:
                return None, True          # too soon

        self._last_fire[trigger] = now

        # Build a tight context string for auto-coach
        context  = self._build_context(stats, exercise)
        extra    = f"\n\nExtra details: {json.dumps(details)}" if details else ""

        prompt = (
            f"Trigger: {trigger}\n"
            f"Sensor data: {json.dumps(context)}"
            f"{extra}"
        )

        messages = [
            {"role": "system",  "content": AUTO_SYSTEM},
            {"role": "user",    "content": prompt},
        ]

        resp  = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.75,
            max_tokens=70,
            top_p=0.9,
        )
        reply = resp.choices[0].message.content.strip()
        return reply, False

    # ── Helpers ───────────────────────────────────────────────────────
    def _build_context(self, stats: dict, exercise: str) -> dict:
        ctx = {
            "exercise":  exercise,
            "has_error": stats.get("has_error", False),
            "feedback":  stats.get("feedback", []),
        }
        if exercise == "squat":
            ctx.update({
                "reps":        stats.get("reps",        0),
                "stage":       stats.get("stage",       "up"),
                "knee_angle":  stats.get("knee_angle",  None),
                "torso_angle": stats.get("torso_angle", None),
            })
        elif exercise == "plank":
            ctx.update({
                "hold_seconds": round(float(stats.get("hold_seconds", 0)), 1),
                "body_angle":   stats.get("body_angle",  None),
                "form_status":  stats.get("form_status", "unknown"),
            })
        return {k: v for k, v in ctx.items() if v is not None}

    def clear_history(self):
        self._history              = []
        self._last_fire            = {}
        self._goal_achieved_fired  = False
