"""Does the preprocessing help a classifier? SVM test on raw vs processed images.

The same hand-made features (HSV colour histograms and LBP texture histograms)
are extracted from four versions of each image:

    A  resize only (the raw image, same size as the others)
    C  CLAHE only
    D  denoise + CLAHE + sharpen, no background mask
    E  full pipeline, with the green background mask

An RBF SVM is trained and tested with stratified 5-fold cross-validation,
repeated 3 times with different shuffles. All versions use the same folds.

Usage:
    python classify_eval.py
    python classify_eval.py --input data/raw --results results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import figures  # sets the plot style
import pest_pipeline as pp

VARIANTS = ["A_resize_only", "C_CLAHE_only", "D_enhanced_no_mask", "E_full_pipeline"]
FOLDS = 5
REPEATS = 3
SEED = 0


def features(bgr: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Colour and texture histograms of the pixels inside `region`."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sel = region > 0
    n = max(1, int(sel.sum()))
    parts = [
        np.bincount(hsv[:, :, 0][sel] // 12, minlength=15)[:15],   # hue, 15 bins
        np.bincount(hsv[:, :, 1][sel] // 32, minlength=8),          # saturation, 8 bins
        np.bincount(hsv[:, :, 2][sel] // 32, minlength=8),          # value, 8 bins
    ]
    for points, radius in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, points, radius, method="uniform").astype(np.int64)
        parts.append(np.bincount(lbp[sel], minlength=points + 2))
    return np.concatenate([p / n for p in parts]).astype(np.float64)


def build_features(input_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    classes = pp.list_classes(input_dir)
    names = list(classes)
    feats: dict[str, list[np.ndarray]] = {v: [] for v in VARIANTS}
    labels: list[int] = []
    for idx, (cls, files) in enumerate(classes.items()):
        for path in files:
            res = pp.process_image(path)
            if res is None:
                continue
            region = pp.content_region(res.content_box, pp.TARGET_SIZE)
            fg_region = cv2.bitwise_and(region, res.final_mask)
            variants = pp.baseline_variants(res.original, res)
            for v in VARIANTS:
                feats[v].append(features(variants[v], fg_region if v == "E_full_pipeline" else region))
            labels.append(idx)
        print(f"  features: {cls}", flush=True)
    return {v: np.vstack(f) for v, f in feats.items()}, np.array(labels), names


def evaluate(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float], np.ndarray]:
    """Return accuracy and macro F1 for each repeat, and the predictions of repeat 0."""
    accs, f1s = [], []
    first_pred = None
    for rep in range(REPEATS):
        pred = np.empty_like(y)
        folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + rep)
        for train, test in folds.split(x, y):
            model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
            model.fit(x[train], y[train])
            pred[test] = model.predict(x[test])
        accs.append(accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro"))
        if rep == 0:
            first_pred = pred.copy()
    return accs, f1s, first_pred


def confusion_figure(path: Path, y: np.ndarray, preds: dict[str, np.ndarray], names: list[str],
                     scores: dict[str, float]) -> None:
    cmap = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#5598e7", "#104281"])
    labels = [figures.pretty(n).replace(" ", "\n") for n in names]
    show = ["A_resize_only", "E_full_pipeline"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, v in zip(axes, show):
        cm = confusion_matrix(y, preds[v])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.sum(axis=1).max())
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                dark = cm[i, j] > 0.55 * cm.sum(axis=1).max()
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="#ffffff" if dark else figures.INK)
        ax.set_xticks(range(len(names)), labels, fontsize=8)
        ax.set_yticks(range(len(names)), labels, fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{pp.VARIANT_LABELS[v].replace(chr(10), ' ')}  (accuracy {100 * scores[v]:.1f}%)")
    fig.tight_layout()
    fig.savefig(path, dpi=figures.DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SVM comparison of raw and processed images")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    (args.results / "figures").mkdir(parents=True, exist_ok=True)

    feats, y, names = build_features(args.input)
    rows, preds, first_acc = [], {}, {}
    per_class_rows = []
    for v in VARIANTS:
        accs, f1s, pred = evaluate(feats[v], y)
        preds[v], first_acc[v] = pred, accs[0]
        rows.append({"variant": v, "images": len(y), "features": feats[v].shape[1],
                     "accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_std": round(float(np.std(accs)), 4),
                     "macro_f1_mean": round(float(np.mean(f1s)), 4), "macro_f1_std": round(float(np.std(f1s)), 4)})
        prec, rec, f1, support = precision_recall_fscore_support(y, pred, zero_division=0)
        for i, name in enumerate(names):
            per_class_rows.append({"variant": v, "class": name, "precision": round(float(prec[i]), 3),
                                   "recall": round(float(rec[i]), 3), "f1": round(float(f1[i]), 3),
                                   "support": int(support[i])})

    pp.write_csv(args.results / "classification_summary.csv", rows)
    pp.write_csv(args.results / "classification_per_class.csv", per_class_rows)
    confusion_figure(args.results / "figures" / "confusion_matrices.png", y, preds, names, first_acc)

    lines = [f"SVM (RBF, C=10), {FOLDS}-fold stratified cross-validation, {REPEATS} repeats, "
             f"{len(y)} images, {len(names)} classes, chance level {100 / len(names):.1f}%",
             f"{'variant':<22}{'accuracy':>18}{'macro F1':>18}"]
    for r in rows:
        lines.append(f"{r['variant']:<22}{100 * r['accuracy_mean']:>10.2f} +/- {100 * r['accuracy_std']:<5.2f}"
                     f"{r['macro_f1_mean']:>10.3f} +/- {r['macro_f1_std']:<5.3f}")
    lines += ["", "Per-class F1 (repeat 0)", f"{'class':<14}" + "".join(f"{v.split('_')[0]:>8}" for v in VARIANTS)]
    for name in names:
        vals = [next(r["f1"] for r in per_class_rows if r["variant"] == v and r["class"] == name) for v in VARIANTS]
        lines.append(f"{name:<14}" + "".join(f"{x:>8.3f}" for x in vals))
    text = "\n".join(lines)
    (args.results / "classification.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
