# Plant Pest Image Preprocessing Pipeline

Batch pipeline that compresses, normalizes and enhances plant pest images in six
classes: green aphid, yellow aphid, black aphid, whitefly, red mite, mealybug.

Stages for each image:

1. Load, scale to 8-bit, reduce to a working resolution.
2. Quality check with the BRISQUE no-reference score. A low-quality image
   (score above 30) is processed by six candidate filters (median, Gaussian,
   Wiener, Richardson-Lucy, total variation, sharpening) and the result with
   the best score is kept. This follows Jia et al. (2025) and makes their
   manual filter selection automatic.
3. Green background mask: HSV green range, then an Otsu threshold on the LAB
   chroma distance from the dominant leaf colour, then morphology.
4. CLAHE on the lightness channel and unsharp masking.
5. Resize with aspect-ratio padding to 512 x 512 and save as JPEG.

Reference paper: Jia, S.; Horri Rezaei, M.; Honarvar Shakibaei Asli, B.
Enhancing Pest Detection in Deep Learning Through a Systematic Image Quality
Assessment and Preprocessing Framework. J. Exp. Theor. Anal. 2025, 3, 39.
https://doi.org/10.3390/jeta3040039

## Setup

Python 3.11 or later.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python run_all.py
```

This command does all the steps with no manual input:

1. `download_dataset.py` downloads the images into `data/raw/<class>/` (skipped
   if they are already there). An internet connection is necessary for this step only.
2. `pest_pipeline.py` writes the processed images to `data/processed/<class>/`
   and the tables and figures to `results/`.
3. `degradation_experiment.py` degrades clean images in a known way and measures
   the PSNR and SSIM of each restoration filter.
4. `classify_eval.py` compares an SVM classifier on raw and processed images.

To use a different dataset, put the images in `data/raw/<class name>/` (one
folder for each class) before you run the command. Each script can also be run
alone; use `--help` to see the options.

## Files

| File | Purpose |
|---|---|
| `run_all.py` | Runs all the steps |
| `download_dataset.py` | Gets the dataset and writes the attribution list |
| `pest_pipeline.py` | The preprocessing pipeline, metrics and summary tables |
| `degradation_experiment.py` | Restoration filter test with reference images |
| `models/` | BRISQUE model files from the OpenCV contrib project |
| `figures.py` | Step plots and charts |
| `classify_eval.py` | SVM comparison of raw and processed images |
| `requirements.txt` | Library versions |

## Outputs

| Path | Content |
|---|---|
| `data/processed/<class>/` | 512 x 512 JPEG images |
| `results/summary.txt` | Storage, time and baseline numbers |
| `results/storage_summary.csv` | Raw MB and processed MB for each class |
| `results/timing_summary.csv` | Mean time for each stage |
| `results/baseline_comparison.csv` | Quality metrics for the baseline methods |
| `results/iqa_*.csv` | BRISQUE before and after restoration, and the filters selected |
| `results/degradation*.{txt,csv}` | PSNR and SSIM of each filter on degraded images |
| `results/metrics_per_image.csv` | All values for each image |
| `results/classification*.{txt,csv}` | Classifier results |
| `results/figures/` | Step plots (Original, HSV Mask, CLAHE Enhanced, Final Resized) and charts |

## Dataset

The images are research-grade observations from iNaturalist with Creative
Commons licences. `data/raw/ATTRIBUTION.csv` lists the photographer, licence and
source page for each image. The same file makes later downloads give the same set.

The images are not stored in this repository. The figures in `results/figures/`
show some of these photos; each photo stays under the licence of its
photographer, as listed in `data/raw/ATTRIBUTION.csv`. Some licences do not
permit commercial use.

## Report

The project report is in `report/report.pdf`. The LaTeX source is in `report/` (compile with `tectonic report.tex`).

## Run the pipeline on your own images

Put images in the folder `run-all-once/` and run:

```
python run_all_once.py
```

The script processes each image in the folder (subfolders also) and writes to `run-all-once/output/`: the final 512 x 512 images, one step plot for each image, `results.csv` and `summary.txt`. Class folders and the dataset download are not necessary. Use `--input <folder>` for a different folder.

## Web page (one image at a time)

```
python web_app.py
```

Then open http://127.0.0.1:8000 in a browser and select an image. The page shows the original and the processed image side by side, the mask, the enhanced image, and the results (size, time for each stage, BRISQUE, selected filter, mask data). The server uses only the Python standard library, listens on this computer only, and saves nothing.
