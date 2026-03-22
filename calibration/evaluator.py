"""
evaluator.py
------------
Applies a set of thresholds to a labeled test DataFrame and produces:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrices
  - Side-by-side comparison plot: original vs trained thresholds
"""

import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
)

_BG  = "#07090e"
_CARD= "#111520"
_MUT = "#4a5a72"
_TXT = "#dde4f0"


def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


# ── Prediction logic ─────────────────────────────────────────────────

def predict_with_thresholds(df: pd.DataFrame, thresholds: dict, mode: str) -> np.ndarray:
    """
    Given a dict of {class: {feature, direction, threshold}} and a
    test DataFrame, return a prediction array (one label per row).

    Logic:
      Squat — check knee_hip_ratio:
        < inward_threshold  → knees_inward
        > outward_threshold → knees_outward   (if threshold exists)
        else                → correct
      Plank — check body_angle:
        < low_threshold  → hips_too_low
        > high_threshold → hips_too_high
        else             → correct
    """
    preds = []

    for _, row in df.iterrows():
        if mode == "squat":
            ratio = row.get("knee_hip_ratio", np.nan)
            pred  = "correct"
            ti    = thresholds.get("knees_inward",  {})
            to    = thresholds.get("knees_outward", {})
            if ti and not np.isnan(ratio) and ratio < ti["threshold"]:
                pred = "knees_inward"
            elif to and not np.isnan(ratio) and ratio > to["threshold"]:
                pred = "knees_outward"
        else:
            angle = row.get("body_angle", np.nan)
            pred  = "correct"
            tl    = thresholds.get("hips_too_low",  {})
            th    = thresholds.get("hips_too_high", {})
            if tl and not np.isnan(angle) and angle < tl["threshold"]:
                pred = "hips_too_low"
            elif th and not np.isnan(angle) and angle > th["threshold"]:
                pred = "hips_too_high"
        preds.append(pred)

    return np.array(preds)


def _safe_metrics(y_true, y_pred, labels):
    """Return per-class precision, recall, F1, support and overall accuracy."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy":  round(acc * 100, 1),
        "per_class": {
            lbl: {
                "precision": round(float(p[i]) * 100, 1),
                "recall":    round(float(r[i]) * 100, 1),
                "f1":        round(float(f[i]) * 100, 1),
                "support":   int(s[i]),
            }
            for i, lbl in enumerate(labels)
        },
    }


# ── Main evaluation entry ─────────────────────────────────────────────

def evaluate(
    test_df:    pd.DataFrame,
    orig_thresholds: dict,
    trained_thresholds: dict,
    mode: str,
    labels_meta: list,
) -> dict:
    """
    Returns a dict with:
      orig_metrics, trained_metrics,
      orig_preds, trained_preds,
      y_true,
      plots: {confusion, comparison, feature_overlay}
    """
    all_labels = [m["id"] for m in labels_meta]
    y_true     = test_df["label"].values

    orig_pred    = predict_with_thresholds(test_df, orig_thresholds,    mode)
    trained_pred = predict_with_thresholds(test_df, trained_thresholds, mode)

    orig_metrics    = _safe_metrics(y_true, orig_pred,    all_labels)
    trained_metrics = _safe_metrics(y_true, trained_pred, all_labels)

    plots = {
        "confusion":       _plot_confusion_matrices(y_true, orig_pred, trained_pred,
                                                     all_labels, labels_meta),
        "comparison":      _plot_metric_comparison(orig_metrics, trained_metrics,
                                                    all_labels, labels_meta),
        "feature_overlay": _plot_feature_overlay(test_df, orig_thresholds,
                                                  trained_thresholds, mode, labels_meta),
    }

    return {
        "orig_metrics":    orig_metrics,
        "trained_metrics": trained_metrics,
        "y_true":          y_true.tolist(),
        "orig_pred":       orig_pred.tolist(),
        "trained_pred":    trained_pred.tolist(),
        "n_test":          len(test_df),
        "plots":           plots,
    }


# ── Plots ─────────────────────────────────────────────────────────────

def _label_colors(labels_meta):
    return {m["id"]: m["color"] for m in labels_meta}


def _plot_confusion_matrices(y_true, orig_pred, trained_pred,
                              labels, labels_meta) -> str:
    colors = _label_colors(labels_meta)
    short  = [l.replace("_", "\n") for l in labels]
    n      = len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_BG)

    for ax, preds, title in [
        (ax1, orig_pred,    "Original\nHardcoded Thresholds"),
        (ax2, trained_pred, "Trained\nCalibrated Thresholds"),
    ]:
        cm  = confusion_matrix(y_true, preds, labels=labels)
        acc = accuracy_score(y_true, preds)

        # draw heatmap manually so we can style it
        ax.set_facecolor(_CARD)
        vmax = cm.max() if cm.max() > 0 else 1
        im   = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax,
                         aspect="auto", interpolation="nearest")

        for i in range(n):
            for j in range(n):
                val  = cm[i, j]
                col  = "white" if val > vmax * 0.5 else _MUT
                ax.text(j, i, str(val), ha="center", va="center",
                        color=col, fontsize=11, fontweight="bold")

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, fontsize=8, color=_MUT)
        ax.set_yticklabels(short, fontsize=8, color=_MUT)
        ax.set_xlabel("Predicted", color=_MUT, fontsize=9)
        ax.set_ylabel("True",      color=_MUT, fontsize=9)
        ax.set_title(f"{title}\nAccuracy: {acc*100:.1f}%",
                     color=_TXT, fontsize=9, pad=10)
        for s in ax.spines.values():
            s.set_color("#1c2535")
        ax.tick_params(colors=_MUT)

    fig.suptitle("Confusion Matrices — Test Set", color=_TXT, fontsize=11, y=1.02)
    fig.tight_layout()
    return _fig_b64(fig)


def _plot_metric_comparison(orig_m, trained_m, labels, labels_meta) -> str:
    colors = _label_colors(labels_meta)
    metrics = ["precision", "recall", "f1"]
    n_lbl   = len(labels)
    n_met   = len(metrics)

    fig, axes = plt.subplots(1, n_met, figsize=(5 * n_met, 4.5))
    fig.patch.set_facecolor(_BG)
    if n_met == 1:
        axes = [axes]

    x     = np.arange(n_lbl)
    width = 0.35

    for ax, met in zip(axes, metrics):
        o_vals = [orig_m["per_class"].get(l, {}).get(met, 0) for l in labels]
        t_vals = [trained_m["per_class"].get(l, {}).get(met, 0) for l in labels]

        bars1 = ax.bar(x - width/2, o_vals, width, label="Original",
                       color="#4a5a72", alpha=0.8, edgecolor="none")
        bars2 = ax.bar(x + width/2, t_vals, width, label="Trained",
                       color=["#00e5a0"] * n_lbl, alpha=0.85, edgecolor="none")

        # colour trained bars per class
        for bar, lbl in zip(bars2, labels):
            bar.set_color(colors.get(lbl, "#00e5a0"))

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel(f"{met.title()} (%)", color=_MUT, fontsize=9)
        ax.set_facecolor(_CARD)
        for s in ax.spines.values():
            s.set_color("#1c2535")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=_MUT, labelsize=8)
        ax.yaxis.label.set_color(_MUT)

        overall_o = orig_m["accuracy"]
        overall_t = trained_m["accuracy"]
        ax.set_title(
            f"{met.title()}\nOrig acc: {overall_o}%  |  Trained acc: {overall_t}%",
            color=_TXT, fontsize=8.5, pad=8,
        )
        ax.legend(facecolor="#161c29", edgecolor="#1c2535", labelcolor=_TXT, fontsize=8)

    fig.tight_layout()
    return _fig_b64(fig)


def _plot_feature_overlay(df: pd.DataFrame, orig_thresh: dict,
                           trained_thresh: dict, mode: str,
                           labels_meta: list) -> str:
    """
    Scatter of the key feature(s) per true label, with vertical/horizontal
    lines for original and trained thresholds.
    """
    colors = _label_colors(labels_meta)

    if mode == "squat":
        feat   = "knee_hip_ratio"
        feat2  = "avg_knee_angle"
        xlabel = "Knee / Hip Width Ratio"
        ylabel = "Avg Knee Angle (°)"
        o_lines = {
            "knees_inward":  ("x", orig_thresh.get("knees_inward",  {}).get("threshold"), "#4a5a72"),
        }
        t_lines = {
            "knees_inward":  ("x", trained_thresh.get("knees_inward",  {}).get("threshold"), "#ff4646"),
            "knees_outward": ("x", trained_thresh.get("knees_outward", {}).get("threshold"), "#3ba7ff"),
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl in df["label"].unique():
            sub = df[df["label"] == lbl]
            ax.scatter(sub[feat], sub[feat2],
                       c=colors.get(lbl, "#666"), s=50, alpha=.82,
                       edgecolors="none",
                       label=next((m["label"] for m in labels_meta if m["id"] == lbl), lbl))
        for cls, (axis, val, col) in o_lines.items():
            if val is not None:
                ax.axvline(val, color=col, lw=1.8, ls="--",
                           label=f"Orig {cls.replace('_',' ')}: {val}")
        for cls, (axis, val, col) in t_lines.items():
            if val is not None:
                ax.axvline(val, color=col, lw=2.2, ls="-",
                           label=f"Trained {cls.replace('_',' ')}: {val}")
        ax.set_xlabel(xlabel, color=_MUT)
        ax.set_ylabel(ylabel, color=_MUT)
        ax.set_title("Test Set — Feature Space + Thresholds", color=_TXT, fontsize=10)

    else:  # plank — single feature
        feat   = "body_angle"
        fig, ax = plt.subplots(figsize=(8, 4))
        for lbl in df["label"].unique():
            vals = df[df["label"] == lbl][feat].values
            jit  = np.random.uniform(-0.2, 0.2, len(vals))
            ax.scatter(
                vals, np.full(len(vals), df["label"].unique().tolist().index(lbl)) + jit,
                c=colors.get(lbl, "#666"), s=50, alpha=.82, edgecolors="none",
                label=next((m["label"] for m in labels_meta if m["id"] == lbl), lbl),
            )
        ulbs = sorted(df["label"].unique())
        ax.set_yticks(range(len(ulbs)))
        ax.set_yticklabels([l.replace("_", "\n") for l in ulbs], fontsize=8)

        o_low  = orig_thresh.get("hips_too_low",  {}).get("threshold")
        o_high = orig_thresh.get("hips_too_high", {}).get("threshold")
        t_low  = trained_thresh.get("hips_too_low",  {}).get("threshold")
        t_high = trained_thresh.get("hips_too_high", {}).get("threshold")
        if o_low:  ax.axvline(o_low,  color="#4a5a72", lw=1.8, ls="--", label=f"Orig low: {o_low}")
        if o_high: ax.axvline(o_high, color="#4a5a72", lw=1.8, ls=":",  label=f"Orig high: {o_high}")
        if t_low:  ax.axvline(t_low,  color="#ff4646", lw=2.2, ls="-",  label=f"Trained low: {t_low}")
        if t_high: ax.axvline(t_high, color="#9d6cff", lw=2.2, ls="-",  label=f"Trained high: {t_high}")
        ax.set_xlabel("Body Angle (°)", color=_MUT)
        ax.set_title("Test Set — Body Angle + Thresholds", color=_TXT, fontsize=10)

    ax.set_facecolor(_CARD)
    for s in ax.spines.values():
        s.set_color("#1c2535")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=_MUT, labelsize=8)
    ax.legend(facecolor="#161c29", edgecolor="#1c2535", labelcolor=_TXT,
              fontsize=7.5, loc="best")
    fig.patch.set_facecolor(_BG)
    fig.tight_layout()
    return _fig_b64(fig)