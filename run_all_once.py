"""Run the full preprocessing pipeline on every image in one folder.

Put images in the folder `run-all-once` (subfolders are also read), then run:

    python run_all_once.py

A different folder can be given:

    python run_all_once.py --input path/to/folder

For each image the script runs all stages of pest_pipeline.py (load, working
resize, BRISQUE quality check and restoration, green background mask, CLAHE,
resize with padding, JPEG) and writes, in `<folder>/output`:

    processed/<name>.jpg     final 512 x 512 image
    steps/<name>_steps.png   Original -> HSV mask -> CLAHE enhanced -> Final resized
    results.csv              size, time, BRISQUE, filter and mask data per image
    summary.txt              totals (raw MB against processed MB, time per image)

No class folders, labels or dataset download are necessary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figures
import pest_pipeline as pp

DEFAULT_FOLDER = Path(__file__).resolve().parent / "run-all-once"
OUTPUT_NAME = "output"


def find_images(folder: Path) -> list[Path]:
    """All image files in the folder and its subfolders, except the output folder."""
    out_dir = folder / OUTPUT_NAME
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in pp.IMAGE_EXTENSIONS
                  and out_dir not in p.parents)


def unique_stem(path: Path, folder: Path, used: set[str]) -> str:
    """Output name. The subfolder is added if two images have the same name."""
    stem = path.stem
    if stem in used:
        stem = "_".join(path.relative_to(folder).with_suffix("").parts)
    n, base = 2, stem
    while stem in used:
        stem, n = f"{base}_{n}", n + 1
    used.add(stem)
    return stem


def save_step_plot(res: pp.Result, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    figures.step_row(axes, res, titles=True)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline on all images in one folder")
    parser.add_argument("--input", type=Path, default=DEFAULT_FOLDER,
                        help="folder with the images (default: run-all-once)")
    parser.add_argument("--no-plots", action="store_true", help="do not write the step plots")
    args = parser.parse_args()

    folder = args.input.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}. Put images in this folder and run the script again.")
        return 1

    out_dir = folder / OUTPUT_NAME
    processed_dir, steps_dir = out_dir / "processed", out_dir / "steps"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        steps_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(images)} images from {folder}", flush=True)
    rows: list[dict] = []
    skipped: list[str] = []
    used: set[str] = set()
    for i, path in enumerate(images, 1):
        rel = str(path.relative_to(folder))
        res = pp.process_image(path)
        if res is None:
            skipped.append(rel)
            print(f"  [{i}/{len(images)}] {rel}: skipped (cannot read the file)", flush=True)
            continue
        stem = unique_stem(path, folder, used)
        t0 = time.perf_counter()
        (processed_dir / f"{stem}.jpg").write_bytes(res.jpeg)
        res.times_ms["save"] += (time.perf_counter() - t0) * 1000.0
        total_ms = sum(res.times_ms.values())

        raw_bytes = path.stat().st_size
        r, m = res.restore, res.mask
        rows.append({
            "file": rel, "output": f"{stem}.jpg",
            "raw_height": res.raw_shape[0], "raw_width": res.raw_shape[1],
            "raw_kb": round(raw_bytes / 1024, 1), "processed_kb": round(len(res.jpeg) / 1024, 1),
            "brisque_before": round(r.brisque_before, 2), "brisque_after": round(r.brisque_after, 2),
            "low_quality": r.flagged, "filter": r.filter,
            "green_fraction": round(m.green_fraction, 3), "foreground_fraction": round(m.fg_fraction, 3),
            "mask_applied": m.applied,
            **{f"{s}_ms": round(res.times_ms[s], 2) for s in pp.STAGES},
            "total_ms": round(total_ms, 2),
        })
        if not args.no_plots:
            note = (f"{rel}   BRISQUE {r.brisque_before:.1f} -> {r.brisque_after:.1f} "
                    f"(filter: {r.filter})   mask {'applied' if m.applied else 'not applied'}")
            save_step_plot(res, note, steps_dir / f"{stem}_steps.png")
        print(f"  [{i}/{len(images)}] {rel}: {total_ms:.0f} ms, BRISQUE {r.brisque_before:.1f} -> "
              f"{r.brisque_after:.1f}, filter {r.filter}, mask {'applied' if m.applied else 'not applied'}",
              flush=True)

    if not rows:
        print("No image could be read.")
        return 1

    pp.write_csv(out_dir / "results.csv", rows)
    n = len(rows)
    raw_mb = sum(r["raw_kb"] for r in rows) / 1024
    proc_mb = sum(r["processed_kb"] for r in rows) / 1024
    times = sorted(r["total_ms"] for r in rows)
    median = times[n // 2] if n % 2 else (times[n // 2 - 1] + times[n // 2]) / 2
    flagged = [r for r in rows if r["low_quality"]]
    lines = [
        f"Images processed       : {n}" + (f" ({len(skipped)} skipped)" if skipped else ""),
        f"Raw size               : {raw_mb:.2f} MB",
        f"Processed size         : {proc_mb:.2f} MB ({100 * (1 - proc_mb / raw_mb):.1f}% less)",
        f"Time per image         : mean {sum(times) / n:.1f} ms, median {median:.1f} ms",
        f"Low quality (BRISQUE > {pp.BRISQUE_THRESHOLD:.0f}): {len(flagged)}",
    ]
    if flagged:
        before = sum(r["brisque_before"] for r in flagged) / len(flagged)
        after = sum(r["brisque_after"] for r in flagged) / len(flagged)
        lines.append(f"  their mean BRISQUE   : {before:.2f} -> {after:.2f}")
    lines += [
        f"Background mask applied: {sum(r['mask_applied'] for r in rows)} of {n}",
        f"Output folder          : {out_dir}",
    ]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
