"""Run the full project from start to end with one command.

    python run_all.py

Steps:
    1. Download the dataset into data/raw (skipped if the images are there).
    2. Run the preprocessing pipeline: data/processed and results/.
    3. Run the controlled degradation test (PSNR and SSIM of each filter).
    4. Run the SVM comparison of raw and processed images.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"


def step(title: str, script: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, check=True)


def main() -> None:
    has_images = RAW.is_dir() and any(p.suffix.lower() == ".jpg" for p in RAW.rglob("*"))
    if has_images:
        print(f"=== 1/4 Dataset found in {RAW}: download skipped ===")
    else:
        step("1/4 Download dataset", "download_dataset.py")
    step("2/4 Preprocessing pipeline", "pest_pipeline.py")
    step("3/4 Controlled degradation test", "degradation_experiment.py")
    step("4/4 Classifier comparison", "classify_eval.py")
    print("\nDone. Processed images: data/processed   Tables and figures: results/")


if __name__ == "__main__":
    main()
