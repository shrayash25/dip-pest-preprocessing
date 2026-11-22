# Plant Pest Image Preprocessing Pipeline

Batch pipeline that compresses, normalizes and enhances plant pest images in six
classes: green aphid, yellow aphid, black aphid, whitefly, red mite, mealybug.

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
3. `classify_eval.py` compares an SVM classifier on raw and processed images.

To use a different dataset, put the images in `data/raw/<class name>/` (one
folder for each class) before you run the command. Each script can also be run
alone; use `--help` to see the options.

## Files

| File | Purpose |
|---|---|
| `run_all.py` | Runs all the steps |
| `download_dataset.py` | Gets the dataset and writes the attribution list |
| `pest_pipeline.py` | The preprocessing pipeline, metrics and summary tables |
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
