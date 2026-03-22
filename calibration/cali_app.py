"""
app.py  —  SmartAssist Calibration Tool
Run:  python calibration/app.py
Opens at http://localhost:5001
"""

import io, os, sys, uuid, threading, json, time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, Response, stream_with_context
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration.pose_engine import extract_video_frames, process_frames
from calibration.features import (
    extract_squat_features, extract_plank_features,
    SQUAT_LABELS, PLANK_LABELS,
    SQUAT_FEAT_COLS, PLANK_FEAT_COLS,
    ORIGINAL_THRESHOLDS,
)
from calibration.trainer import (
    run_svm, compute_thresholds,
    plot_scatter, plot_threshold, plot_confusion_train,
    generate_snippet,
)
from calibration.evaluator import evaluate

TMP = Path("calib_tmp"); TMP.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

TRAIN_FRAMES: list[dict] = []
TEST_FRAMES:  list[dict] = []
MODE: str = "squat"
_last_trained_thresholds: dict = {}

# ── Progress tracking (shared across threads) ─────────────────────────
_progress_lock = threading.Lock()
_progress = {"current": 0, "total": 0, "kept": 0, "skipped": 0, "done": True, "msg": ""}


def _set_progress(current, total, kept, skipped):
    with _progress_lock:
        _progress.update(current=current, total=total,
                         kept=kept, skipped=skipped, done=False)


def _finish_progress():
    with _progress_lock:
        _progress["done"] = True


def _frames(dataset):
    return TRAIN_FRAMES if dataset == "train" else TEST_FRAMES

def _feat_fn():
    return extract_squat_features if MODE == "squat" else extract_plank_features

def _labels():
    return SQUAT_LABELS if MODE == "squat" else PLANK_LABELS

def _feat_cols():
    return SQUAT_FEAT_COLS if MODE == "squat" else PLANK_FEAT_COLS

def _fr(fr):
    return {"id": fr["id"], "img": fr["img_b64"],
            "detected": fr["detected"], "label": fr["label"]}


# ── Pages ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Progress SSE ──────────────────────────────────────────────────────
@app.route("/api/progress")
def api_progress():
    """Server-Sent Events endpoint — client reads this while processing."""
    def generate():
        while True:
            with _progress_lock:
                p = dict(_progress)
            yield f"data: {json.dumps(p)}\n\n"
            if p["done"]:
                break
            time.sleep(0.25)
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Mode ──────────────────────────────────────────────────────────────
@app.route("/api/mode")
def get_mode():
    return jsonify({"mode": MODE, "labels": _labels()})

@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    global MODE, TRAIN_FRAMES, TEST_FRAMES, _last_trained_thresholds
    MODE                     = (request.get_json() or {}).get("mode", "squat")
    TRAIN_FRAMES             = []
    TEST_FRAMES              = []
    _last_trained_thresholds = {}
    return jsonify({"mode": MODE, "labels": _labels()})


# ── Frame collection ──────────────────────────────────────────────────
@app.route("/api/process_video", methods=["POST"])
def api_process_video():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No file received"}), 400
    dataset  = request.form.get("dataset", "train")
    interval = float(request.form.get("interval", 0.2))
    suffix   = Path(f.filename or "v.mp4").suffix or ".mp4"
    tmp      = TMP / f"{uuid.uuid4().hex}{suffix}"
    f.save(tmp)
    try:
        raw, fps, duration = extract_video_frames(tmp, interval)
        if not raw:
            return jsonify({"error": "No frames extracted — bad video file?"}), 400

        with _progress_lock:
            _progress.update(current=0, total=len(raw), kept=0,
                             skipped=0, done=False, msg="Running MediaPipe…")

        new, skipped = process_frames(raw, _feat_fn(), progress_cb=_set_progress)
        _finish_progress()
        _frames(dataset).extend(new)

        return jsonify({
            "added": len(new), "skipped": skipped,
            "total": len(new) + skipped, "dataset": dataset,
            "fps": round(fps, 1), "duration": round(duration, 1),
            "frames": [_fr(f) for f in new],
        })
    finally:
        tmp.unlink(missing_ok=True)


@app.route("/api/process_images", methods=["POST"])
def api_process_images():
    files   = request.files.getlist("images")
    dataset = request.form.get("dataset", "train")
    if not files:
        return jsonify({"error": "No images received"}), 400
    raw = []
    for f in files:
        buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            raw.append(img)
    if not raw:
        return jsonify({"error": "No valid images decoded"}), 400

    with _progress_lock:
        _progress.update(current=0, total=len(raw), kept=0,
                         skipped=0, done=False, msg="Running MediaPipe…")

    new, skipped = process_frames(raw, _feat_fn(), progress_cb=_set_progress)
    _finish_progress()
    _frames(dataset).extend(new)

    return jsonify({"added": len(new), "skipped": skipped,
                    "dataset": dataset, "frames": [_fr(f) for f in new]})


# ── Labelling ─────────────────────────────────────────────────────────
@app.route("/api/bulk_label", methods=["POST"])
def api_bulk_label():
    d       = request.get_json() or {}
    ids     = set(d.get("ids", []))
    lbl     = d.get("label")
    dataset = d.get("dataset", "train")
    for fr in _frames(dataset):
        if fr["id"] in ids:
            fr["label"] = lbl
    return jsonify({"ok": True, "updated": len(ids)})

@app.route("/api/delete_frame", methods=["POST"])
def api_delete_frame():
    global TRAIN_FRAMES, TEST_FRAMES
    d       = request.get_json() or {}
    fid     = d.get("id")
    dataset = d.get("dataset", "train")
    if dataset == "train":
        TRAIN_FRAMES = [f for f in TRAIN_FRAMES if f["id"] != fid]
    else:
        TEST_FRAMES = [f for f in TEST_FRAMES if f["id"] != fid]
    return jsonify({"ok": True})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    global TRAIN_FRAMES, TEST_FRAMES, _last_trained_thresholds
    dataset = (request.get_json() or {}).get("dataset", "all")
    if dataset in ("train", "all"):
        TRAIN_FRAMES = []; _last_trained_thresholds = {}
    if dataset in ("test", "all"):
        TEST_FRAMES = []
    return jsonify({"ok": True})

@app.route("/api/stats")
def api_stats():
    def _s(frames):
        by = {}
        for f in frames:
            if f["label"]: by[f["label"]] = by.get(f["label"], 0) + 1
        return {"total": len(frames), "labeled": sum(1 for f in frames if f["label"]), "by_label": by}
    return jsonify({"train": _s(TRAIN_FRAMES), "test": _s(TEST_FRAMES)})


# ── Training ──────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def api_train():
    global _last_trained_thresholds
    labeled = [fr for fr in TRAIN_FRAMES if fr["label"] and fr["features"]]
    if len(labeled) < 6:
        return jsonify({"error": f"Need ≥ 6 labeled TRAIN frames. Have {len(labeled)}."}), 400
    df                       = pd.DataFrame([{**fr["features"], "label": fr["label"]} for fr in labeled])
    svm_res, svm_err         = run_svm(df, _feat_cols())
    thresholds               = compute_thresholds(df, MODE)
    _last_trained_thresholds = thresholds
    code                     = generate_snippet(thresholds, MODE, svm_res)
    lm                       = _labels()
    return jsonify({
        "svm": svm_res, "svm_error": svm_err,
        "thresholds": thresholds, "code": code,
        "n_labeled": len(labeled),
        "label_counts": df["label"].value_counts().to_dict(),
        "plots": {
            "scatter":    plot_scatter(df, MODE, lm),
            "threshold":  plot_threshold(df, thresholds, lm),
            "confusion":  plot_confusion_train(df, thresholds, MODE, lm),
        },
    })


# ── Comparison ────────────────────────────────────────────────────────
@app.route("/api/compare", methods=["POST"])
def api_compare():
    labeled_test = [fr for fr in TEST_FRAMES if fr["label"] and fr["features"]]
    if len(labeled_test) < 3:
        return jsonify({"error": f"Need ≥ 3 labeled TEST frames. Have {len(labeled_test)}."}), 400
    if not _last_trained_thresholds:
        return jsonify({"error": "Train the model first (Step 3) before comparing."}), 400
    test_df     = pd.DataFrame([{**fr["features"], "label": fr["label"]} for fr in labeled_test])
    orig_thresh = ORIGINAL_THRESHOLDS.get(MODE, {})
    lm          = _labels()
    result      = evaluate(test_df, orig_thresh, _last_trained_thresholds, MODE, lm)
    result["original_thresholds"] = {
        k: {"threshold": v["threshold"], "direction": v["direction"],
            "feature": v["feature"], "note": v.get("note", "")}
        for k, v in orig_thresh.items()
    }
    result["trained_thresholds"] = {
        k: {"threshold": v["threshold"], "direction": v["direction"], "feature": v["feature"]}
        for k, v in _last_trained_thresholds.items()
    }
    return jsonify(result)


# ── CSV export ────────────────────────────────────────────────────────
@app.route("/api/export_csv")
def api_export_csv():
    dataset = request.args.get("dataset", "train")
    frames  = _frames(dataset)
    labeled = [fr for fr in frames if fr["label"] and fr["features"]]
    if not labeled:
        return jsonify({"error": "No labeled frames to export"}), 400
    df  = pd.DataFrame([{**fr["features"], "label": fr["label"], "frame_id": fr["id"]} for fr in labeled])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename={MODE}_{dataset}.csv"})


if __name__ == "__main__":
    print("\n" + "=" * 54)
    print("  SmartAssist — Calibration Tool")
    print("  Open  http://localhost:5001")
    print("=" * 54 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)