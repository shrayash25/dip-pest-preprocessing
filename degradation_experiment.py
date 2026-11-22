"""Controlled test of the restoration filters with a known reference image.

BRISQUE needs no reference image, but it is also the score that selects the
filter, so it cannot prove that the selected filter is correct. This test
uses clean images as the reference:

    1. Take clean images (BRISQUE <= CLEAN_MAX) from each class.
    2. Degrade each one in three ways: blur, Gaussian noise, salt-and-pepper noise.
    3. Apply each restoration filter.
    4. Measure PSNR and SSIM against the clean image, and BRISQUE.
    5. Compare the filter that BRISQUE selects with the filter that has the
       best PSNR.

Usage:
    python degradation_experiment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import figures
import pest_pipeline as pp

CLEAN_MAX = 20.0        # BRISQUE limit for a reference image
PER_CLASS = 10
SIDE = 768              # long side used in this test
SEED = 0

BLUR_SIGMA = 1.5
NOISE_SIGMA = 15.0      # grey levels
SALT_PEPPER = 0.04      # fraction of pixels

DEGRADATIONS = ["blur", "gaussian_noise", "salt_pepper"]
DEG_LABELS = {"blur": f"Gaussian blur (sigma {BLUR_SIGMA})",
              "gaussian_noise": f"Gaussian noise (sigma {NOISE_SIGMA:.0f})",
              "salt_pepper": f"Salt-and-pepper noise ({100 * SALT_PEPPER:.0f}%)"}


def degrade(clean: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "blur":
        out = cv2.GaussianBlur(clean, (0, 0), BLUR_SIGMA).astype(np.float32)
        out += rng.normal(0.0, 2.0, out.shape)             # small sensor noise
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "gaussian_noise":
        out = clean.astype(np.float32) + rng.normal(0.0, NOISE_SIGMA, clean.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    out = clean.copy()
    r = rng.random(clean.shape[:2])
    out[r < SALT_PEPPER / 2] = 0
    out[r > 1 - SALT_PEPPER / 2] = 255
    return out


def psnr_ssim(clean: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    psnr = peak_signal_noise_ratio(clean, img, data_range=255)
    ssim = structural_similarity(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), data_range=255)
    return float(psnr), float(ssim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restoration filter test with reference images")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    filters = pp.RESTORE_FILTERS
    records: list[dict] = []
    example: dict[str, dict[str, np.ndarray]] = {}
    n_clean = 0
    for cls, files in pp.list_classes(args.input).items():
        taken = 0
        for path in files:
            if taken == PER_CLASS:
                break
            raw = pp.load_image(path)
            if raw is None:
                continue
            clean = pp.limit_long_side(raw, SIDE)
            if pp.brisque_score(clean) > CLEAN_MAX:
                continue
            taken += 1
            n_clean += 1
            for kind in DEGRADATIONS:
                bad = degrade(clean, kind, rng)
                cands = pp.restoration_candidates(bad)
                rec = {"class": cls, "file": path.name, "degradation": kind}
                for name in filters:
                    psnr, ssim = psnr_ssim(clean, cands[name])
                    rec[f"psnr_{name}"], rec[f"ssim_{name}"] = psnr, ssim
                    rec[f"brisque_{name}"] = pp.brisque_score(cands[name])
                rec["selected"] = min(filters, key=lambda n: rec[f"brisque_{n}"])
                rec["best_psnr"] = max(filters, key=lambda n: rec[f"psnr_{n}"])
                records.append(rec)
                if kind not in example and cls == "yellow_aphid":
                    example[kind] = {"clean": clean, "degraded": bad,
                                     "selected": cands[rec["selected"]], "best": cands[rec["best_psnr"]],
                                     "selected_name": rec["selected"], "best_name": rec["best_psnr"],
                                     "psnr": {n: rec[f"psnr_{n}"] for n in filters}}
        print(f"  {cls}: {taken} clean images", flush=True)

    # ---- tables ------------------------------------------------------------
    filter_rows, summary_rows = [], []
    for kind in DEGRADATIONS:
        recs = [r for r in records if r["degradation"] == kind]
        for name in filters:
            filter_rows.append({
                "degradation": kind, "filter": name,
                "mean_psnr_db": round(float(np.mean([r[f"psnr_{name}"] for r in recs])), 2),
                "mean_ssim": round(float(np.mean([r[f"ssim_{name}"] for r in recs])), 4),
                "mean_brisque": round(float(np.mean([r[f"brisque_{name}"] for r in recs])), 2),
                "selected_by_brisque": sum(r["selected"] == name for r in recs),
                "best_by_psnr": sum(r["best_psnr"] == name for r in recs),
            })
        summary_rows.append({
            "degradation": kind, "images": len(recs),
            "psnr_degraded": round(float(np.mean([r["psnr_none"] for r in recs])), 2),
            "psnr_brisque_selected": round(float(np.mean([r[f"psnr_{r['selected']}"] for r in recs])), 2),
            "psnr_best_filter": round(float(np.mean([r[f"psnr_{r['best_psnr']}"] for r in recs])), 2),
            "ssim_degraded": round(float(np.mean([r["ssim_none"] for r in recs])), 4),
            "ssim_brisque_selected": round(float(np.mean([r[f"ssim_{r['selected']}"] for r in recs])), 4),
            "ssim_best_filter": round(float(np.mean([r[f"ssim_{r['best_psnr']}"] for r in recs])), 4),
            "selection_agrees_percent": round(100.0 * float(np.mean([r["selected"] == r["best_psnr"] for r in recs])), 1),
        })
    pp.write_csv(args.results / "degradation_filters.csv", filter_rows)
    pp.write_csv(args.results / "degradation_summary.csv", summary_rows)

    # ---- chart: PSNR of each filter for each degradation ---------------------
    labels = [("Degraded" if n == "none" else figures.FILTER_LABELS[n]).replace("-", "-\n").replace(" ", "\n")
              for n in filters]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    for ax, kind in zip(axes, DEGRADATIONS):
        rows = [r for r in filter_rows if r["degradation"] == kind]
        bars = ax.bar(labels, [r["mean_psnr_db"] for r in rows], width=0.62, color=figures.BLUE)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=figures.INK_2, padding=2)
        ax.set_title(DEG_LABELS[kind])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8)
    axes[0].set_ylabel("Mean PSNR (dB), higher is better")
    fig.tight_layout()
    fig.savefig(fig_dir / "degradation_psnr.png", dpi=figures.DPI)
    plt.close(fig)

    # ---- visual example ------------------------------------------------------
    if example:
        fig, axes = plt.subplots(len(example), 4, figsize=(11, 2.9 * len(example)), squeeze=False)
        for i, kind in enumerate(k for k in DEGRADATIONS if k in example):
            ex = example[kind]
            h, w = ex["clean"].shape[:2]
            side = min(h, w) // 2
            y0, x0 = (h - side) // 2, (w - side) // 2
            crop = lambda im: im[y0:y0 + side, x0:x0 + side]
            name = lambda n: "no change" if n == "none" else figures.FILTER_LABELS[n]
            figures.show(axes[i][0], crop(ex["clean"]), "Clean reference" if i == 0 else None)
            figures.show(axes[i][1], crop(ex["degraded"]))
            axes[i][1].set_title(f"{DEG_LABELS[kind]}\nPSNR {ex['psnr']['none']:.1f} dB", fontsize=9, loc="center")
            figures.show(axes[i][2], crop(ex["selected"]))
            axes[i][2].set_title(f"Selected by BRISQUE: {name(ex['selected_name'])}\n"
                                 f"PSNR {ex['psnr'][ex['selected_name']]:.1f} dB", fontsize=9, loc="center")
            figures.show(axes[i][3], crop(ex["best"]))
            axes[i][3].set_title(f"Best PSNR: {name(ex['best_name'])}\n"
                                 f"PSNR {ex['psnr'][ex['best_name']]:.1f} dB", fontsize=9, loc="center")
        fig.tight_layout()
        fig.savefig(fig_dir / "degradation_examples.png", dpi=figures.DPI)
        plt.close(fig)

    # ---- text summary --------------------------------------------------------
    lines = [f"Controlled degradation test: {n_clean} clean reference images (BRISQUE <= {CLEAN_MAX:.0f})",
             "", "Mean PSNR in dB (and how often BRISQUE selected the filter / how often it had the best PSNR)",
             f"{'filter':<18}" + "".join(f"{k:>26}" for k in DEGRADATIONS)]
    for name in filters:
        cells = []
        for kind in DEGRADATIONS:
            r = next(x for x in filter_rows if x["degradation"] == kind and x["filter"] == name)
            cells.append(f"{r['mean_psnr_db']:>12.2f} ({r['selected_by_brisque']:>2}/{r['best_by_psnr']:>2})".rjust(26))
        lines.append(f"{name:<18}" + "".join(cells))
    lines += ["", f"{'degradation':<18}{'PSNR degraded':>15}{'BRISQUE choice':>16}{'best filter':>13}"
                  f"{'SSIM degraded':>15}{'BRISQUE choice':>16}{'best filter':>13}{'agree %':>9}"]
    for r in summary_rows:
        lines.append(f"{r['degradation']:<18}{r['psnr_degraded']:>15.2f}{r['psnr_brisque_selected']:>16.2f}"
                     f"{r['psnr_best_filter']:>13.2f}{r['ssim_degraded']:>15.4f}{r['ssim_brisque_selected']:>16.4f}"
                     f"{r['ssim_best_filter']:>13.4f}{r['selection_agrees_percent']:>9.1f}")
    text = "\n".join(lines)
    (args.results / "degradation.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
