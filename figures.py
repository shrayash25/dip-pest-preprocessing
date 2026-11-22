"""Figures for the pest preprocessing pipeline (step plots and charts)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pest_pipeline as pp

# Colours
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})
DPI = 200


def rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pretty(cls: str) -> str:
    return cls.replace("_", " ").title()


def show(ax, img, title: str | None = None, gray: bool = False) -> None:
    if gray:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(rgb(img))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="center")


STEP_TITLES = ["Original", "HSV mask (white = kept)", "CLAHE enhanced", "Final resized"]


def step_row(axes, res: pp.Result, titles: bool) -> None:
    show(axes[0], res.original, STEP_TITLES[0] if titles else None)
    show(axes[1], res.mask.foreground, STEP_TITLES[1] if titles else None, gray=True)
    show(axes[2], res.enhanced, STEP_TITLES[2] if titles else None)
    show(axes[3], res.final, STEP_TITLES[3] if titles else None)


def steps_per_class(fig_dir: Path, samples: dict) -> None:
    for cls, items in samples.items():
        if not items:
            continue
        fig, axes = plt.subplots(len(items), 4, figsize=(11, 2.9 * len(items)), squeeze=False)
        for r, (_, res) in enumerate(items):
            step_row(axes[r], res, titles=(r == 0))
        fig.suptitle(f"Processing steps: {pretty(cls)}", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / f"steps_{cls}.png", dpi=DPI)
        plt.close(fig)


def steps_overview(fig_dir: Path, samples: dict) -> None:
    classes = [c for c, items in samples.items() if items]
    fig, axes = plt.subplots(len(classes), 4, figsize=(10, 2.5 * len(classes)), squeeze=False)
    for r, cls in enumerate(classes):
        _, res = samples[cls][len(samples[cls]) // 2]
        step_row(axes[r], res, titles=(r == 0))
        axes[r][0].set_ylabel(pretty(cls), fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "steps_overview.png", dpi=DPI)
    plt.close(fig)


def storage_chart(fig_dir: Path, storage_rows: list[dict]) -> None:
    rows = [r for r in storage_rows if r["class"] != "TOTAL"]
    total = next(r for r in storage_rows if r["class"] == "TOTAL")
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    raw = ax.bar(x - width / 2 - 0.01, [r["raw_MB"] for r in rows], width, color=BLUE, label="Raw")
    proc = ax.bar(x + width / 2 + 0.01, [r["processed_MB"] for r in rows], width, color=ORANGE, label="Processed")
    ax.bar_label(raw, fmt="%.1f", fontsize=8, color=INK_2, padding=2)
    ax.bar_label(proc, fmt="%.1f", fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x, [pretty(r["class"]) for r in rows])
    ax.set_ylabel("Storage (MB)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper right", ncols=2)
    ax.set_title(f"Storage for each class: {total['raw_MB']:.1f} MB raw to "
                 f"{total['processed_MB']:.1f} MB processed ({total['reduction_percent']:.1f}% less)")
    fig.tight_layout()
    fig.savefig(fig_dir / "storage_per_class.png", dpi=DPI)
    plt.close(fig)


def timing_chart(fig_dir: Path, timing_rows: list[dict]) -> None:
    rows = [r for r in timing_rows if r["stage"] != "TOTAL"]
    total = next(r for r in timing_rows if r["stage"] == "TOTAL")
    labels = {"load": "Load + decode", "work_resize": "Working resize", "denoise": "Denoise",
              "mask": "Green mask", "enhance": "CLAHE + sharpen", "final_resize": "Final resize + pad",
              "save": "JPEG encode + write"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(rows))[::-1]
    bars = ax.barh(y, [r["mean_ms"] for r in rows], height=0.55, color=BLUE)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_2, padding=3)
    ax.set_yticks(y, [labels[r["stage"]] for r in rows])
    ax.set_xlabel("Mean time per image (ms)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(f"Time for each stage (total {total['mean_ms']:.1f} ms per image)")
    fig.tight_layout()
    fig.savefig(fig_dir / "time_per_stage.png", dpi=DPI)
    plt.close(fig)


def baseline_chart(fig_dir: Path, baseline_summary: list[dict]) -> None:
    names = [r["variant"] for r in baseline_summary]
    short = [n.split("_")[0] for n in names]
    panels = [("mean_contrast", "RMS contrast", "%.1f"), ("mean_entropy", "Entropy (bits)", "%.2f"),
              ("mean_sharpness", "Sharpness (Laplacian variance)", "%.0f"),
              ("total_MB", "Total size (MB)", "%.1f")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (key, title, fmt) in zip(axes, panels):
        bars = ax.bar(short, [r[key] for r in baseline_summary], width=0.6, color=BLUE)
        ax.bar_label(bars, fmt=fmt, fontsize=8, color=INK_2, padding=2)
        ax.set_title(title)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    caption = "   ".join(pp.VARIANT_LABELS[n].replace("\n", " ") for n in names)
    fig.text(0.01, 0.01, caption, fontsize=8, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_dir / "baseline_metrics.png", dpi=DPI)
    plt.close(fig)


def baseline_visual(fig_dir: Path, samples: dict) -> None:
    classes = [c for c, items in samples.items() if items]
    names = list(pp.VARIANT_LABELS)
    fig, axes = plt.subplots(len(classes), len(names), figsize=(2.3 * len(names), 2.4 * len(classes)),
                             squeeze=False)
    for r, cls in enumerate(classes):
        _, res = samples[cls][0]
        variants = pp.baseline_variants(res.original, res)
        for c, name in enumerate(names):
            show(axes[r][c], variants[name], pp.VARIANT_LABELS[name] if r == 0 else None)
        axes[r][0].set_ylabel(pretty(cls), fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_visual.png", dpi=DPI)
    plt.close(fig)


def clahe_histogram(fig_dir: Path, samples: dict) -> None:
    cls = next(c for c, items in samples.items() if items)
    _, res = samples[cls][0]
    before = cv2.cvtColor(res.original, cv2.COLOR_BGR2LAB)[:, :, 0]
    he = cv2.cvtColor(pp.global_hist_equalize(res.original), cv2.COLOR_BGR2LAB)[:, :, 0]
    after = cv2.cvtColor(pp.clahe_on_lightness(res.original, sharpen=False), cv2.COLOR_BGR2LAB)[:, :, 0]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), height_ratios=[1.5, 1])
    for c, (img, title) in enumerate([(before, "Original L channel"),
                                      (he, "Global histogram equalization"),
                                      (after, f"CLAHE (clip {pp.CLAHE_CLIP}, grid {pp.CLAHE_GRID[0]}x{pp.CLAHE_GRID[1]})")]):
        show(axes[0][c], img, title, gray=True)
        ax = axes[1][c]
        hist = np.bincount(img.ravel(), minlength=256) / img.size
        ax.fill_between(np.arange(256), hist, color=BLUE, linewidth=0)
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("L value")
        if c == 0:
            ax.set_ylabel("Fraction of pixels")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(fig_dir / "clahe_histogram.png", dpi=DPI)
    plt.close(fig)


def quality_chart(fig_dir: Path, per_image: list[dict]) -> None:
    classes = sorted({r["class"] for r in per_image}, key=[r["class"] for r in per_image].index)
    panels = [("contrast", "RMS contrast"), ("entropy", "Entropy (bits)"),
              ("sharpness", "Sharpness (Laplacian variance)")]
    x = np.arange(len(classes))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (key, title) in zip(axes, panels):
        before = [np.mean([r[f"before_{key}"] for r in per_image if r["class"] == c]) for c in classes]
        after = [np.mean([r[f"after_{key}"] for r in per_image if r["class"] == c]) for c in classes]
        ax.bar(x - width / 2 - 0.01, before, width, color=BLUE, label="Before (resize only)")
        ax.bar(x + width / 2 + 0.01, after, width, color=ORANGE, label="After (full pipeline)")
        ax.set_xticks(x, [pretty(c).replace(" ", "\n") for c in classes], fontsize=8)
        ax.set_title(title)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "quality_before_after.png", dpi=DPI)
    plt.close(fig)


def foreground_chart(fig_dir: Path, per_image: list[dict]) -> None:
    classes = sorted({r["class"] for r in per_image}, key=[r["class"] for r in per_image].index)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = [[100 * r["fg_fraction"] for r in per_image if r["class"] == c] for c in classes]
    box = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops={"color": INK, "linewidth": 1.5},
                     flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                                 "markeredgecolor": "none"},
                     whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for patch in box["boxes"]:
        patch.set(facecolor=BLUE, edgecolor="none", alpha=0.75)
    ax.set_xticks(np.arange(1, len(classes) + 1), [pretty(c) for c in classes])
    ax.set_ylabel("Foreground kept (% of pixels)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Share of each image kept after green background masking")
    fig.tight_layout()
    fig.savefig(fig_dir / "foreground_fraction.png", dpi=DPI)
    plt.close(fig)


def failure_cases(fig_dir: Path, per_image: list[dict], input_dir: Path) -> None:
    """Show the images where masking removed the most and the least."""
    applied = [r for r in per_image if r["mask_applied"]]
    fallback = [r for r in per_image if not r["mask_applied"]]
    picks: list[tuple[str, dict]] = []
    for r in sorted(applied, key=lambda r: r["fg_fraction"])[:2]:
        picks.append((f"Most removed: {100 * r['fg_fraction']:.0f}% kept", r))
    for r in sorted(applied, key=lambda r: -r["fg_fraction"])[:2]:
        picks.append((f"Least removed: {100 * r['fg_fraction']:.0f}% kept", r))
    for r in sorted(fallback, key=lambda r: r["fg_fraction"])[:2]:
        picks.append((f"Mask not applied ({100 * r['fg_fraction']:.0f}% foreground)", r))
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 4, figsize=(10, 2.6 * len(picks)), squeeze=False)
    for i, (label, r) in enumerate(picks):
        res = pp.process_image(input_dir / r["class"] / r["file"])
        if res is None:
            continue
        step_row(axes[i], res, titles=(i == 0))
        axes[i][0].set_ylabel(f"{pretty(r['class'])}\n{label}", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "mask_extreme_cases.png", dpi=DPI)
    plt.close(fig)


def make_all(fig_dir: Path, samples: dict, per_image: list[dict], storage_rows: list[dict],
             timing_rows: list[dict], baseline_summary: list[dict], input_dir: Path) -> None:
    steps_per_class(fig_dir, samples)
    steps_overview(fig_dir, samples)
    storage_chart(fig_dir, storage_rows)
    timing_chart(fig_dir, timing_rows)
    baseline_chart(fig_dir, baseline_summary)
    baseline_visual(fig_dir, samples)
    clahe_histogram(fig_dir, samples)
    quality_chart(fig_dir, per_image)
    foreground_chart(fig_dir, per_image)
    failure_cases(fig_dir, per_image, input_dir)
