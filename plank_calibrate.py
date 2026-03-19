"""
plank_calibrate.py  —  SmartAssist
=====================================
Calibrates plank thresholds from YOUR body + YOUR camera angle.

Usage:
    python plank_calibrate.py

Camera placement matters a lot:
  ✅ BEST  — camera to your SIDE, waist height, ~2m away
  ⚠️  OK   — camera slightly elevated to side
  ❌ BAD   — camera in front of you (body angle always looks flat)
"""

import time, os, urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

MODEL_PATH = "pose_landmarker_lite.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model…")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task", MODEL_PATH)

L_SHOULDER, R_SHOULDER = 11, 12
L_HIP,      R_HIP      = 23, 24
L_ANKLE,    R_ANKLE    = 27, 28
NOSE                   = 0
SAMPLE_SECONDS         = 7

# Quality gates — frames outside these are thrown away as bad detections
MIN_VALID_ANGLE = 100.0   # below this = MediaPipe lost the body
MAX_VALID_ANGLE = 240.0   # above this = also a detection artifact
MIN_QUALITY_FRAMES = 20   # need at least this many clean frames to trust a pose
MAX_ACCEPTABLE_STD = 15.0  # body angle std above this = bad camera or unstable

def make_landmarker():
    opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return PoseLandmarker.create_from_options(opts)

def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def angle_3pts(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def px(lm, idx, w, h):
    l = lm[idx]; return (l.x*w, l.y*h)

def extract_features(lm, w, h):
    ls=px(lm,L_SHOULDER,w,h); rs=px(lm,R_SHOULDER,w,h)
    lh=px(lm,L_HIP,w,h);      rh=px(lm,R_HIP,w,h)
    la=px(lm,L_ANKLE,w,h);    ra=px(lm,R_ANKLE,w,h)
    ns=px(lm,NOSE,w,h)
    ms=midpoint(ls,rs); mh=midpoint(lh,rh); ma=midpoint(la,ra)
    body_ang  = angle_3pts(ms, mh, ma)
    nose_frac = (ms[1]-ns[1])/h
    hip_frac  = abs(lh[1]-rh[1])/h
    return body_ang, nose_frac, hip_frac

def collect_pose(cap, landmarker, label, color, duration=SAMPLE_SECONDS):
    all_samples   = []
    clean_samples = []
    state   = "waiting"
    t_start = None

    print(f"\n{'='*54}")
    print(f"  POSE: {label}")
    print(f"  Get into position, then press SPACE to start.")
    print(f"  Hold steady for {duration} seconds.")
    print(f"{'='*54}")

    while True:
        ok, frame = cap.read()
        if not ok: continue
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        detected = bool(result.pose_landmarks)

        # Draw skeleton
        if detected:
            lms = result.pose_landmarks[0]
            for a,b in [(L_SHOULDER,L_HIP),(R_SHOULDER,R_HIP),
                        (L_HIP,L_ANKLE),(R_HIP,R_ANKLE),
                        (L_SHOULDER,R_SHOULDER),(L_HIP,R_HIP)]:
                pa=px(lms,a,w,h); pb=px(lms,b,w,h)
                cv2.line(frame,(int(pa[0]),int(pa[1])),(int(pb[0]),int(pb[1])),(80,200,255),2)
            for idx in [L_SHOULDER,R_SHOULDER,L_HIP,R_HIP,L_ANKLE,R_ANKLE]:
                p=px(lms,idx,w,h)
                cv2.circle(frame,(int(p[0]),int(p[1])),5,(0,255,180),-1)

        # Detection warning
        if not detected:
            cv2.rectangle(frame,(0,0),(w,40),(0,0,80),-1)
            cv2.putText(frame,"⚠ Body not detected — move into frame",(10,28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,120,255),2)

        if state == "waiting":
            cv2.putText(frame,f"POSE: {label}",(14,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
            cv2.putText(frame,"Press SPACE when in position",(14,104),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        elif state == "recording":
            elapsed = time.time()-t_start
            remain  = max(0, duration-elapsed)
            pct     = int(elapsed/duration*w)
            cv2.rectangle(frame,(0,h-8),(pct,h),color,-1)
            cv2.putText(frame,f"RECORDING — {remain:.1f}s left",(14,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

            if detected:
                feats = extract_features(result.pose_landmarks[0], w, h)
                all_samples.append(feats)
                valid = MIN_VALID_ANGLE <= feats[0] <= MAX_VALID_ANGLE
                if valid:
                    clean_samples.append(feats)
                quality_col = (0,255,120) if valid else (0,80,255)
                cv2.putText(frame,
                    f"body={feats[0]:.1f}  {'OK' if valid else 'BAD FRAME — discarded'}",
                    (14,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.52,quality_col,1)

            # Live quality counter
            cv2.putText(frame,
                f"clean frames: {len(clean_samples)} / {len(all_samples)}",
                (14,104),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1)

            if elapsed >= duration:
                state = "done"

        elif state == "done":
            n_clean = len(clean_samples)
            n_total = len(all_samples)
            msg = f"Done — {n_clean}/{n_total} clean frames. Press any key."
            col = (0,255,180) if n_clean>=MIN_QUALITY_FRAMES else (0,80,255)
            cv2.putText(frame,msg,(14,70),cv2.FONT_HERSHEY_SIMPLEX,0.65,col,2)

        cv2.imshow("SmartAssist — Plank Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if state=="waiting" and key==ord(" "):
            state="recording"; t_start=time.time(); all_samples=[]; clean_samples=[]
        elif state=="done" and key!=255:
            break
        elif key==ord("q"):
            print("Cancelled."); return None

    return clean_samples

def quality_check(samples, label):
    if not samples:
        print(f"\n  ❌ {label}: NO clean frames captured.")
        return None
    arr = np.array(samples)
    mean, std = arr[:,0].mean(), arr[:,0].std()
    n = len(arr)
    print(f"\n  {label}  ({n} clean frames)")
    print(f"    body_angle : mean={mean:.1f}  std={std:.1f}  "
          f"min={arr[:,0].min():.1f}  max={arr[:,0].max():.1f}")
    print(f"    nose_frac  : mean={arr[:,1].mean():.3f}  std={arr[:,1].std():.3f}")
    print(f"    hip_frac   : mean={arr[:,2].mean():.3f}  std={arr[:,2].std():.3f}")

    if n < MIN_QUALITY_FRAMES:
        print(f"    ⚠️  Only {n} clean frames — need {MIN_QUALITY_FRAMES}. Redo this pose.")
        return None
    if std > MAX_ACCEPTABLE_STD:
        print(f"    ⚠️  Angle std={std:.1f}° is too high (>{MAX_ACCEPTABLE_STD}°).")
        print(f"       This usually means the camera isn't side-on, or you were moving.")
        print(f"       Try placing camera to your LEFT or RIGHT side at waist height.")
        return None
    print(f"    ✅ Quality OK")
    return arr

def compute_thresholds(good, sag, pike):
    good_mean = good[:,0].mean()
    good_std  = good[:,0].std()

    sag_max  = sag[:,0].max()   if sag  is not None else good_mean - 25
    pike_min = pike[:,0].min()  if pike is not None else good_mean + 25

    # Leave a margin beyond the bad-pose boundary
    angle_min = round(min(good_mean - 2.5*good_std, sag_max  + 3), 1)
    angle_max = round(max(good_mean + 2.5*good_std, pike_min - 3), 1)
    angle_min = max(130.0, angle_min)
    angle_max = min(230.0, angle_max)

    nose_hi  = round(good[:,1].mean() + 2.5*good[:,1].std(), 3)
    nose_lo  = round(good[:,1].mean() - 2.5*good[:,1].std(), 3)
    hip_ceil = round(good[:,2].mean() + 2.5*good[:,2].std(), 3)

    # Safety clamps — don't make nose/hip thresholds absurdly large
    nose_hi  = min(nose_hi,  0.30)
    nose_lo  = max(nose_lo, -0.20)
    hip_ceil = min(hip_ceil, 0.25)

    return dict(angle_min=angle_min, angle_max=angle_max,
                nose_hi=nose_hi, nose_lo=nose_lo, hip_ceil=hip_ceil)

def print_result(t, good_std):
    # If camera was front-facing, body angle is unreliable — flag it
    angle_range = t['angle_max'] - t['angle_min']
    front_camera_warning = (angle_range > 80)

    print("\n" + "="*54)
    print("  CALIBRATION RESULT")
    print("="*54)

    if front_camera_warning:
        print("""
  ⚠️  WARNING: Body angle range is very wide ({:.0f}°).
  This usually means the camera is front-facing.
  The body angle check will be unreliable from this angle.
  For best results, place the camera to your SIDE.
  The nose/hip checks will still work from any angle.
""".format(angle_range))

    print("  Paste these into exercises/plank.py:\n")
    print(f"BODY_ANGLE_MIN = {t['angle_min']}   # hips sagging threshold")
    print(f"BODY_ANGLE_MAX = {t['angle_max']}   # hips piking threshold")
    print(f"""
  Form check block:

if body_angle < BODY_ANGLE_MIN:
    errors.append(f"Hips sagging ({{body_angle:.0f}}°) — engage core & raise hips")
elif body_angle > BODY_ANGLE_MAX:
    errors.append(f"Hips too high ({{body_angle:.0f}}°) — lower hips to a straight line")

nose_above_shoulder = mid_shoulder[1] - nose[1]
if nose_above_shoulder > h * {t['nose_hi']}:
    errors.append("Head too high — look at the floor ahead of your hands")
elif nose_above_shoulder < -h * {abs(t['nose_lo'])}:
    errors.append("Head dropping — keep neck neutral")

hip_diff = abs(l_hip[1] - r_hip[1])
if hip_diff > h * {t['hip_ceil']}:
    errors.append("Hips are tilted — keep them level")
""")
    print("="*54)

def main():
    print("\n" + "="*54)
    print("  SmartAssist — Plank Calibration Tool")
    print("="*54)
    print("""
You will record 3 poses (6 seconds each):
  1. YOUR CORRECT plank
  2. Hips sagging (back curved down)
  3. Hips piked   (bum in the air)

IMPORTANT — Camera placement:
  ✅  Place camera to your SIDE (left or right), waist height
  ✅  Make sure shoulder → hip → ankle are all visible
  ❌  Do NOT use a front-facing camera for this

Frames where MediaPipe loses your body are automatically
discarded — you'll see "BAD FRAME" on screen.

Press ENTER to start.
""")
    input()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    config = [
        ("CORRECT PLANK", (0,255,180)),
        ("HIPS SAGGING",  (30,100,255)),
        ("HIPS PIKED",    (0,160,255)),
    ]
    raw = {}
    with make_landmarker() as lm:
        for label, color in config:
            data = collect_pose(cap, lm, label, color)
            if data is None:
                cap.release(); cv2.destroyAllWindows(); return
            raw[label] = data

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*54)
    print("  DATA SUMMARY")
    print("="*54)
    good = quality_check(raw["CORRECT PLANK"], "CORRECT PLANK")
    sag  = quality_check(raw["HIPS SAGGING"],  "HIPS SAGGING")
    pike = quality_check(raw["HIPS PIKED"],    "HIPS PIKED")

    if good is None:
        print("\n  ❌ Cannot calibrate without clean correct-plank data.")
        print("  Fix: ensure camera is to your SIDE and your full body is visible.")
        return

    t = compute_thresholds(good, sag, pike)
    print_result(t, good[:,0].std())

if __name__ == "__main__":
    main()
