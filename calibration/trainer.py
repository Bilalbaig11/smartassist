"""
trainer.py
----------
SVM cross-validation, optimal 1-D threshold sweep, matplotlib plots,
and Python code-snippet generation.
"""

import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Colour palette (matches main project dark theme) ─────────────────
_BG   = "#07090e"
_CARD = "#111520"
_MUT  = "#4a5a72"
_TXT  = "#dde4f0"


def _label_colors(labels_meta):
    return {m["id"]: m["color"] for m in labels_meta}


def _fig_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def _style_ax(ax, title=""):
    ax.set_facecolor(_CARD)
    for s in ax.spines.values():
        s.set_color("#1c2535")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=_MUT, labelsize=8)
    ax.xaxis.label.set_color(_MUT)
    ax.yaxis.label.set_color(_MUT)
    if title:
        ax.set_title(title, color=_TXT, fontsize=9, pad=8)


# ── Threshold sweep ───────────────────────────────────────────────────

def find_threshold_1d(vals: np.ndarray, binary: np.ndarray):
    """
    Sweep every midpoint between sorted unique values.
    Returns (threshold, direction '<' or '>', accuracy_fraction).
    """
    sv = np.sort(np.unique(vals))
    if len(sv) < 2:
        return float(sv[0]) if len(sv) else 0.0, ">", 0.5
    mids = (sv[:-1] + sv[1:]) / 2.0
    best = (0.5, mids[0], ">")
    for t in mids:
        for d in ("<", ">"):
            pred = (vals < t) if d == "<" else (vals > t)
            acc  = float(np.mean(pred.astype(int) == binary))
            if acc > best[0]:
                best = (acc, t, d)
    return best[1], best[2], best[0]


def compute_thresholds(df: pd.DataFrame, mode: str) -> dict:
    """
    For each error class, find the best 1-D threshold on its primary feature.
    Squat: knee_hip_ratio separates inward/outward.
    Plank: body_angle separates hips_too_low / hips_too_high.
    """
    if mode == "squat":
        mapping = {
            "knees_inward":  "knee_hip_ratio",
            "knees_outward": "knee_hip_ratio",
        }
    else:
        mapping = {
            "hips_too_low":  "body_angle",
            "hips_too_high": "body_angle",
        }

    out = {}
    for cls, feat in mapping.items():
        if cls not in df["label"].values:
            continue
        sub = df[df["label"].isin(["correct", cls])].copy()
        if len(sub) < 4:
            continue
        binary = (sub["label"] == cls).astype(int).values
        vals   = sub[feat].values
        t, d, acc = find_threshold_1d(vals, binary)
        out[cls] = {
            "feature":      feat,
            "threshold":    round(float(t), 3),
            "direction":    d,
            "accuracy":     round(acc * 100, 1),
            "correct_mean": round(float(df[df["label"] == "correct"][feat].mean()), 3),
            "error_mean":   round(float(df[df["label"] == cls][feat].mean()), 3),
            "n_correct":    int((sub["label"] == "correct").sum()),
            "n_error":      int((sub["label"] == cls).sum()),
        }
    return out


# ── SVM ───────────────────────────────────────────────────────────────

def run_svm(df: pd.DataFrame, feat_cols: list):
    fc = [c for c in feat_cols if c in df.columns]
    X  = df[fc].values
    y  = df["label"].values

    counts = pd.Series(y).value_counts()
    valid  = counts[counts >= 2].index.tolist()
    mask   = df["label"].isin(valid)
    X, y   = X[mask], y[mask]

    if len(np.unique(y)) < 2 or len(X) < 6:
        return None, "Need ≥ 2 classes with ≥ 2 samples each."

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("svm", SVC(kernel="rbf", C=5, gamma="scale",
                    class_weight="balanced", probability=True)),
    ])
    k  = min(5, int(counts[valid].min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    sc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    pipe.fit(X, y)

    return {
        "cv_acc":    round(sc.mean() * 100, 1),
        "cv_std":    round(sc.std()  * 100, 1),
        "labels":    sorted(np.unique(y).tolist()),
        "n":         int(len(X)),
        "feat_cols": fc,
    }, None


# ── Plots ─────────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, mode: str, labels_meta: list) -> str:
    colors_map = _label_colors(labels_meta)
    c = [colors_map.get(l, "#666") for l in df["label"]]

    if mode == "squat":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.patch.set_facecolor(_BG)

        # Plot 1: knee angle vs ratio
        ax1.scatter(df["avg_knee_angle"], df["knee_hip_ratio"],
                    c=c, s=55, alpha=.85, edgecolors="none")
        ax1.set_xlabel("Avg Knee Angle (°)")
        ax1.set_ylabel("Knee / Hip Width Ratio")
        _style_ax(ax1, "Depth  vs  Knee Alignment")

        # Plot 2: L vs R knee angle symmetry
        ax2.scatter(df["l_knee_angle"], df["r_knee_angle"],
                    c=c, s=55, alpha=.85, edgecolors="none")
        ax2.plot([60, 180], [60, 180], color="#2e3d52", lw=1, ls="--")
        ax2.set_xlabel("Left Knee Angle (°)")
        ax2.set_ylabel("Right Knee Angle (°)")
        _style_ax(ax2, "Left  vs  Right Symmetry")

    else:  # plank — single strip plot
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))
        fig.patch.set_facecolor(_BG)
        ulbs = sorted(df["label"].unique())
        for i, lbl in enumerate(ulbs):
            vals = df[df["label"] == lbl]["body_angle"].values
            jit  = np.random.uniform(-0.18, 0.18, len(vals))
            ax1.scatter(
                np.full(len(vals), i) + jit, vals,
                c=colors_map.get(lbl, "#666"), s=52, alpha=.85,
                edgecolors="none",
                label=next((m["label"] for m in labels_meta if m["id"] == lbl), lbl),
            )
        ax1.set_xticks(range(len(ulbs)))
        ax1.set_xticklabels([l.replace("_", "\n") for l in ulbs], fontsize=8)
        ax1.set_ylabel("Body Angle (°)")
        ax1.axhline(180, color="#2e3d52", lw=1, ls="--", label="180° reference")
        _style_ax(ax1, "Body Angle by Class")

    # shared legend
    handles = [
        mpatches.Patch(color=colors_map.get(m["id"], "#666"), label=m["label"])
        for m in labels_meta if m["id"] in df["label"].values
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               facecolor="#161c29", edgecolor="#1c2535", labelcolor=_TXT, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return _fig_b64(fig, dpi=150)


def plot_threshold(df: pd.DataFrame, thresholds: dict, labels_meta: list) -> str | None:
    if not thresholds:
        return None
    items = list(thresholds.items())
    n     = len(items)
    colors_map = _label_colors(labels_meta)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0))
    fig.patch.set_facecolor(_BG)
    if n == 1:
        axes = [axes]

    for ax, (cls, info) in zip(axes, items):
        feat = info["feature"]
        sub  = df[df["label"].isin(["correct", cls])]
        for lbl in ["correct", cls]:
            vals = sub[sub["label"] == lbl][feat].dropna().values
            y    = np.random.uniform(-0.2, 0.2, len(vals))
            meta = next((m for m in labels_meta if m["id"] == lbl), {})
            ax.scatter(vals, y, c=meta.get("color", "#666"), s=46, alpha=.85,
                       label=meta.get("label", lbl), edgecolors="none", zorder=5)
        ax.axvline(info["threshold"], color="#ff8c42", lw=2.5, ls="--",
                   label=f"Threshold: {info['threshold']}")
        ax.set_yticks([])
        ax.set_xlabel(feat.replace("_", " ").title(), color=_MUT, fontsize=9)
        ax.legend(fontsize=7.5, facecolor="#161c29", edgecolor="#1c2535", labelcolor=_TXT)
        _style_ax(ax, f"{cls.replace('_', ' ').title()}  ·  {info['accuracy']}% accuracy")

    fig.tight_layout()
    return _fig_b64(fig, dpi=150)


# ── Code snippet ──────────────────────────────────────────────────────

def generate_snippet(thresholds: dict, mode: str, svm_info: dict | None) -> str:
    lines = [
        f"# ── Generated by SmartAssist calibration tool ───────────────",
        f"# Mode: {mode.upper()}",
    ]
    if svm_info:
        lines.append(
            f"# SVM cross-val: {svm_info['cv_acc']}% ± {svm_info['cv_std']}%"
            f"  ({svm_info['n']} samples)"
        )
    lines.append("")

    if mode == "squat":
        lines.append("# Paste into exercises/squat.py")
        lines.append("")
        ti = thresholds.get("knees_inward",  {}).get("threshold")
        to = thresholds.get("knees_outward", {}).get("threshold")

        for cls, info in thresholds.items():
            lines.append(
                f"# {cls.upper()}: {info['feature']} {info['direction']} "
                f"{info['threshold']}  ({info['accuracy']}% accurate, "
                f"correct_mean={info['correct_mean']}, error_mean={info['error_mean']})"
            )

        lines.append("")
        if ti is not None and to is not None:
            lo, hi = sorted([ti, to])
            lines += [
                f"KNEE_INWARD_RATIO  = {lo}   # knee/hip ratio below this → caving in",
                f"KNEE_OUTWARD_RATIO = {hi}   # knee/hip ratio above this → too wide",
                "",
                "# Replace the knee valgus check in SquatAnalyzer.analyze():",
                "knee_width = abs(l_knee[0] - r_knee[0])",
                "hip_width  = abs(l_hip[0]  - r_hip[0])",
                "ratio = knee_width / (hip_width + 1e-8)",
                "if ratio < KNEE_INWARD_RATIO:",
                "    errors.append('Knees caving in — push them out over toes')",
                "elif ratio > KNEE_OUTWARD_RATIO:",
                "    errors.append('Knees too wide — bring them closer')",
            ]
        elif ti is not None:
            lines += [
                f"KNEE_INWARD_RATIO = {ti}",
                "if knee_width / (hip_width + 1e-8) < KNEE_INWARD_RATIO:",
                "    errors.append('Knees caving in — push them out over toes')",
            ]
        elif to is not None:
            lines += [
                f"KNEE_OUTWARD_RATIO = {to}",
                "if knee_width / (hip_width + 1e-8) > KNEE_OUTWARD_RATIO:",
                "    errors.append('Knees too wide — bring them closer')",
            ]

    else:  # plank
        lines.append("# Paste into exercises/plank.py")
        lines.append("")
        for cls, info in thresholds.items():
            lines.append(
                f"# {cls.upper()}: body_angle {info['direction']} "
                f"{info['threshold']}  ({info['accuracy']}% accurate, "
                f"correct_mean={info['correct_mean']}, error_mean={info['error_mean']})"
            )
        lines.append("")
        if "hips_too_low" in thresholds:
            lines.append(
                f"BODY_ANGLE_MIN = {thresholds['hips_too_low']['threshold']}"
                "   # below = hips sagging"
            )
        if "hips_too_high" in thresholds:
            lines.append(
                f"BODY_ANGLE_MAX = {thresholds['hips_too_high']['threshold']}"
                "   # above = hips piking"
            )
        lines += [
            "",
            "# The existing check in PlankAnalyzer.analyze() already uses these:",
            "# if body_angle < BODY_ANGLE_MIN: → hips sagging",
            "# elif body_angle > BODY_ANGLE_MAX: → hips too high",
        ]

    return "\n".join(lines)


# ── Confusion matrix on training set ─────────────────────────────────

def plot_confusion_train(df: pd.DataFrame, thresholds: dict,
                         mode: str, labels_meta: list) -> str | None:
    """
    Apply the computed thresholds to the training DataFrame itself and
    produce a confusion matrix — gives a quick in-sample sanity check.
    """
    from calibration.evaluator import predict_with_thresholds
    from sklearn.metrics import confusion_matrix, accuracy_score

    all_labels = [m["id"] for m in labels_meta]
    present    = [l for l in all_labels if l in df["label"].values]
    if len(present) < 2:
        return None

    y_true = df["label"].values
    y_pred = predict_with_thresholds(df, thresholds, mode)
    cm     = confusion_matrix(y_true, y_pred, labels=present)
    acc    = accuracy_score(y_true, y_pred)
    colors = _label_colors(labels_meta)
    n      = len(present)
    short  = [l.replace("_", "\n") for l in present]

    fig, ax = plt.subplots(figsize=(max(5, n * 2), max(4.5, n * 1.8)))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)

    vmax = cm.max() if cm.max() > 0 else 1
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax,
              aspect="auto", interpolation="nearest")

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            col = "white" if val > vmax * 0.5 else _MUT
            ax.text(j, i, str(val), ha="center", va="center",
                    color=col, fontsize=13, fontweight="bold")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=9, color=_MUT)
    ax.set_yticklabels(short, fontsize=9, color=_MUT)
    ax.set_xlabel("Predicted", color=_MUT, fontsize=10)
    ax.set_ylabel("True",      color=_MUT, fontsize=10)
    ax.set_title(f"Training Set — Confusion Matrix\nIn-sample accuracy: {acc*100:.1f}%",
                 color=_TXT, fontsize=10, pad=12)
    for s in ax.spines.values():
        s.set_color("#1c2535")
    ax.tick_params(colors=_MUT)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data