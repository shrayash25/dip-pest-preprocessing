"""Image preprocessing and feature enhancement pipeline for plant pest images.

Stages for each image:
    1. Load and scale the bit depth to 8-bit, 3-channel BGR.
    2. Reduce to a working resolution (long side <= WORK_SIDE).
    3. Quality check and restoration: the BRISQUE no-reference quality score
       is calculated. A good image is not changed. A low-quality image is
       processed by each candidate filter (median, Gaussian, Wiener,
       Richardson-Lucy, total variation, sharpening) and the result with the
       best score is kept.
    4. Mask the green leaf background: HSV green range, then an Otsu
       threshold on the LAB chroma distance from the dominant leaf colour,
       then cleaned by morphology.
    5. Enhance: CLAHE on the LAB lightness channel, then unsharp masking.
    6. Resize with aspect-ratio padding to TARGET_SIZE x TARGET_SIZE.
    7. Save as a compressed 8-bit JPEG in a folder for the pest class.

Outputs:
    <output>/<class>/*.jpg          processed images
    <results>/metrics_per_image.csv per-image sizes, times and quality values
    <results>/storage_summary.csv   raw MB vs processed MB for each class
    <results>/timing_summary.csv    mean time for each stage
    <results>/baseline_comparison.csv
    <results>/iqa_class_summary.csv BRISQUE before and after for each class
    <results>/iqa_filter_summary.csv  how often each restoration filter was kept
    <results>/figures/*.png         step plots and charts
    <results>/summary.txt           the numbers printed at the end of the run

Usage:
    python pest_pipeline.py
    python pest_pipeline.py --input data/raw --output data/processed --results results
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TARGET_SIZE = 512          # final image is TARGET_SIZE x TARGET_SIZE
WORK_SIDE = 1024           # long side used for masking and enhancement
JPEG_QUALITY = 85

MEDIAN_KSIZE = 3           # median filter window (impulse noise)
GAUSS_SIGMA = 1.0          # Gaussian filter (sensor noise)

MASK_BLUR_SIGMA = 2.0      # smoothing before thresholding, to reduce speckle
HSV_GREEN_LOW = (35, 30, 40)     # OpenCV scale: H 0-179, S 0-255, V 0-255
HSV_GREEN_HIGH = (90, 255, 255)
MIN_GREEN_FRACTION = 0.02  # below this there is no leaf background to remove
DIST_T_MIN = 8             # limits for the Otsu chroma-distance threshold
DIST_T_MAX = 25            # (8-bit LAB units)
MIN_FOREGROUND = 0.01      # below this the mask is not applied (pest would be lost)
MAX_FOREGROUND = 0.98      # above this there is no green background to remove
OPEN_KSIZE = 5
CLOSE_KSIZE = 9
DILATE_KSIZE = 7           # safety margin kept around the foreground
MIN_REGION_FRACTION = 0.0003   # connected regions smaller than this are removed

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
UNSHARP_SIGMA = 1.0
UNSHARP_AMOUNT = 0.6

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BRISQUE_THRESHOLD = 30.0   # images with a higher score get restoration (Jia et al., 2025)
PSF_SIGMA = 1.2            # Gaussian point spread function assumed for deblurring
WIENER_K = 0.02            # noise-to-signal power ratio for the Wiener filter
RL_ITERATIONS = 10         # Richardson-Lucy iterations
TV_WEIGHT = 0.10           # total variation denoising weight
RESTORE_SHARPEN_AMOUNT = 1.0
MODEL_DIR = Path(__file__).resolve().parent / "models"

STAGES = ["load", "work_resize", "iqa_restore", "mask", "enhance", "final_resize", "save"]
RESTORE_FILTERS = ["none", "median", "gaussian", "wiener", "richardson_lucy", "total_variation", "sharpen"]


# --------------------------------------------------------------------------
# Processing functions
# --------------------------------------------------------------------------
def load_image(path: Path) -> np.ndarray | None:
    """Read an image and return 8-bit, 3-channel BGR (bit-depth scaling)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = np.round(img / 257.0).astype(np.uint8)       # 16-bit -> 8-bit
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def limit_long_side(img: np.ndarray, side: int) -> np.ndarray:
    """Shrink the image so that its long side is at most `side` pixels."""
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Fast noise standard deviation estimate (Immerkaer, 1996)."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    response = cv2.filter2D(gray.astype(np.float64), -1, kernel)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) * np.abs(response).sum() / (6.0 * (w - 2) * (h - 2)))


_brisque_model = None


def brisque_score(bgr: np.ndarray) -> float:
    """BRISQUE no-reference quality score. Lower is better (0 to about 100)."""
    global _brisque_model
    if _brisque_model is None:
        _brisque_model = cv2.quality.QualityBRISQUE_create(
            str(MODEL_DIR / "brisque_model_live.yml"), str(MODEL_DIR / "brisque_range_live.yml"))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(_brisque_model.compute(gray)[0])


def gaussian_psf(sigma: float) -> np.ndarray:
    size = 2 * int(np.ceil(3 * sigma)) + 1
    k = cv2.getGaussianKernel(size, sigma)
    return (k @ k.T).astype(np.float64)


def wiener_deconvolve(l: np.ndarray, sigma: float = PSF_SIGMA, k: float = WIENER_K) -> np.ndarray:
    """Wiener filter in the frequency domain: F = conj(H) / (|H|^2 + K) * G.

    `l` is a float image in [0, 1]. K is the noise-to-signal power ratio.
    """
    psf = gaussian_psf(sigma)
    pad = psf.shape[0]
    g = cv2.copyMakeBorder(l, pad, pad, pad, pad, cv2.BORDER_REFLECT)   # reduces edge ringing
    h = np.zeros_like(g, dtype=np.float64)
    ph, pw = psf.shape
    h[:ph, :pw] = psf
    h = np.roll(h, (-(ph // 2), -(pw // 2)), axis=(0, 1))               # centre the PSF at (0, 0)
    H = np.fft.rfft2(h)
    G = np.fft.rfft2(g)
    F = np.conj(H) / (np.abs(H) ** 2 + k) * G
    f = np.fft.irfft2(F, s=g.shape)
    return np.clip(f[pad:-pad, pad:-pad], 0.0, 1.0).astype(np.float32)


def richardson_lucy(l: np.ndarray, sigma: float = PSF_SIGMA, iterations: int = RL_ITERATIONS) -> np.ndarray:
    """Richardson-Lucy iterative deconvolution with a Gaussian PSF.

    f_(k+1) = f_k * ( h_flipped (*) ( g / (h (*) f_k) ) ). The Gaussian PSF is
    symmetric, so the flipped PSF is the same as the PSF.
    """
    g = l.astype(np.float32)
    estimate = g.copy()
    for _ in range(iterations):
        predicted = cv2.GaussianBlur(estimate, (0, 0), sigma)
        ratio = g / (predicted + 1e-6)
        estimate *= cv2.GaussianBlur(ratio, (0, 0), sigma)
    return np.clip(estimate, 0.0, 1.0)


def total_variation(bgr: np.ndarray, weight: float = TV_WEIGHT) -> np.ndarray:
    """Total variation denoising (Chambolle): smooths noise and keeps edges.

    Noise is in all three colour channels, so this filter uses the full
    colour image and not only the lightness channel.
    """
    from skimage.restoration import denoise_tv_chambolle
    out = denoise_tv_chambolle(bgr.astype(np.float32) / 255.0, weight=weight, channel_axis=-1)
    return np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def sharpen_filter(l: np.ndarray, amount: float = RESTORE_SHARPEN_AMOUNT) -> np.ndarray:
    """Unsharp masking: adds back the high-frequency part of the image."""
    blur = cv2.GaussianBlur(l, (0, 0), UNSHARP_SIGMA)
    return np.clip(l + amount * (l - blur), 0.0, 1.0).astype(np.float32)


def restoration_candidates(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Apply each restoration filter.

    The smoothing filters (median, Gaussian, total variation) use the colour
    image. The deblurring filters (Wiener, Richardson-Lucy, sharpening) use
    the LAB lightness channel, so that they do not make colour fringes.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32) / 255.0
    restored_l = {
        "wiener": wiener_deconvolve(l),
        "richardson_lucy": richardson_lucy(l),
        "sharpen": sharpen_filter(l),
    }
    out = {"none": bgr,
           "median": cv2.medianBlur(bgr, MEDIAN_KSIZE),
           "gaussian": cv2.GaussianBlur(bgr, (0, 0), GAUSS_SIGMA),
           "total_variation": total_variation(bgr)}
    for name, channel in restored_l.items():
        merged = lab.copy()
        merged[:, :, 0] = np.round(channel * 255.0).astype(np.uint8)
        out[name] = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


@dataclass
class RestoreInfo:
    brisque_before: float
    brisque_after: float
    filter: str                    # the filter that was kept
    flagged: bool                  # True if the score was above BRISQUE_THRESHOLD
    scores: dict[str, float]       # score of each candidate (flagged images only)
    noise_sigma: float


def iqa_guided_restore(bgr: np.ndarray, keep_candidates: bool = False):
    """Quality check, then restoration chosen by the quality score.

    Returns (restored image, RestoreInfo) and, if keep_candidates is True,
    also the dict of candidate images.
    """
    noise_sigma = estimate_noise_sigma(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    before = brisque_score(bgr)
    if before <= BRISQUE_THRESHOLD:              # good quality: no change
        info = RestoreInfo(before, before, "none", False, {}, noise_sigma)
        return (bgr, info, {}) if keep_candidates else (bgr, info)
    candidates = restoration_candidates(bgr)
    scores = {"none": before}
    scores.update({name: brisque_score(img) for name, img in candidates.items() if name != "none"})
    best = min(scores, key=scores.get)
    info = RestoreInfo(before, scores[best], best, True, scores, noise_sigma)
    return (candidates[best], info, candidates) if keep_candidates else (candidates[best], info)


def otsu_threshold(channel: np.ndarray) -> tuple[int, float]:
    """Otsu threshold of an 8-bit channel.

    Returns (t, eta). Pixels <= t are class 0. eta is the separability
    measure sigma_between^2 / sigma_total^2 in [0, 1]; a value near 1 means
    two well separated classes.
    """
    hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.zeros(256)
    valid = denom > 1e-12
    sigma_b[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denom[valid]
    t = int(np.argmax(sigma_b))
    sigma_total = float(np.sum((levels - mu_total) ** 2 * p))
    eta = float(sigma_b[t] / sigma_total) if sigma_total > 1e-12 else 0.0
    return t, eta


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected white regions that have fewer than `min_area` pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


@dataclass
class MaskInfo:
    foreground: np.ndarray     # uint8, 255 = keep, 0 = green background
    hsv_green: np.ndarray      # uint8, 255 = pixel in the HSV green range
    otsu_t: int                # Otsu threshold of the chroma distance
    otsu_eta: float            # Otsu separability measure
    dist_t: int                # threshold used, after the limits are applied
    green_fraction: float      # fraction of pixels in the HSV green range
    fg_fraction: float         # foreground fraction before the fallback
    applied: bool              # False when the fallback kept the full image


def histogram_median(values: np.ndarray) -> float:
    """Median of 8-bit values from the histogram (faster than sorting)."""
    cdf = np.cumsum(np.bincount(values, minlength=256))
    return float(np.searchsorted(cdf, cdf[-1] / 2.0))


def green_background_mask(bgr: np.ndarray) -> MaskInfo:
    """Find the green leaf background and return the foreground mask.

    1. HSV range: candidate leaf pixels are those with a green hue.
    2. LAB: the dominant leaf colour is the median (a*, b*) of the candidates.
       The chroma distance of each pixel from that colour is calculated.
    3. Otsu: a threshold on the chroma distance splits the candidates into
       "leaf" (near the dominant colour) and "other" (a pest with a green
       tint, a vein, a highlight). Limits keep the threshold in a safe range.
    4. Morphology and region filtering clean the mask.
    """
    smooth = cv2.GaussianBlur(bgr, (0, 0), MASK_BLUR_SIGMA)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)

    hsv_green = cv2.inRange(hsv, HSV_GREEN_LOW, HSV_GREEN_HIGH)
    candidates = hsv_green > 0
    green_fraction = float(np.count_nonzero(candidates)) / candidates.size
    full = np.full_like(hsv_green, 255)
    if green_fraction < MIN_GREEN_FRACTION:        # no leaf background to remove
        return MaskInfo(full, hsv_green, 0, 0.0, 0, green_fraction, 1.0, False)

    a_channel, b_channel = lab[:, :, 1], lab[:, :, 2]
    a_med = histogram_median(a_channel[candidates])
    b_med = histogram_median(b_channel[candidates])
    dist = np.sqrt((a_channel.astype(np.float32) - a_med) ** 2 +
                   (b_channel.astype(np.float32) - b_med) ** 2)
    dist = np.clip(dist, 0, 255).astype(np.uint8)

    otsu_t, eta = otsu_threshold(dist[candidates])
    dist_t = int(np.clip(otsu_t, DIST_T_MIN, DIST_T_MAX))
    green = np.where(candidates & (dist <= dist_t), 255, 0).astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KSIZE, OPEN_KSIZE))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE, CLOSE_KSIZE))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KSIZE, DILATE_KSIZE))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k_open)     # remove specks
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k_close)   # fill small holes

    min_area = int(MIN_REGION_FRACTION * green.size)
    foreground = remove_small_regions(cv2.bitwise_not(green), min_area)   # leaf texture specks
    foreground = cv2.bitwise_not(remove_small_regions(cv2.bitwise_not(foreground), min_area))  # holes
    foreground = cv2.dilate(foreground, k_dilate)               # safety margin

    fg_fraction = float(np.count_nonzero(foreground)) / foreground.size
    applied = MIN_FOREGROUND <= fg_fraction <= MAX_FOREGROUND
    if not applied:
        foreground = full
    return MaskInfo(foreground, hsv_green, otsu_t, eta, dist_t, green_fraction, fg_fraction, applied)


def clahe_on_lightness(bgr: np.ndarray, sharpen: bool = True) -> np.ndarray:
    """CLAHE on the LAB L channel, optionally followed by unsharp masking."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    if sharpen:
        blur = cv2.GaussianBlur(l, (0, 0), UNSHARP_SIGMA)
        l = cv2.addWeighted(l, 1.0 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def global_hist_equalize(bgr: np.ndarray) -> np.ndarray:
    """Baseline method: global histogram equalization of the L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((cv2.equalizeHist(l), a, b)), cv2.COLOR_LAB2BGR)


def letterbox(img: np.ndarray, size: int, nearest: bool = False) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize keeping the aspect ratio, then pad with zeros to size x size.

    Returns the padded image and the content box (top, left, height, width).
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if nearest:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    shape = (size, size) + img.shape[2:]
    out = np.zeros(shape, dtype=img.dtype)
    out[top:top + nh, left:left + nw] = resized
    return out, (top, left, nh, nw)


def encode_jpeg(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
                                         cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


@dataclass
class Result:
    original: np.ndarray       # working-resolution input
    mask: MaskInfo
    enhanced: np.ndarray       # after CLAHE and sharpening, before masking
    final: np.ndarray          # TARGET_SIZE x TARGET_SIZE output
    final_mask: np.ndarray     # foreground mask at the final size
    content_box: tuple[int, int, int, int]
    restored: np.ndarray       # after the quality-guided restoration
    restore: RestoreInfo
    times_ms: dict[str, float]
    jpeg: bytes
    raw_shape: tuple[int, int]


def process_image(path: Path) -> Result | None:
    """Run all stages on one image. Returns None if the file is not an image."""
    times: dict[str, float] = {}

    t0 = time.perf_counter()
    raw = load_image(path)
    times["load"] = time.perf_counter() - t0
    if raw is None:
        return None

    t0 = time.perf_counter()
    work = limit_long_side(raw, WORK_SIDE)
    times["work_resize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clean, restore = iqa_guided_restore(work)
    times["iqa_restore"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask = green_background_mask(clean)
    times["mask"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enhanced = clahe_on_lightness(clean, sharpen=True)
    masked = cv2.bitwise_and(enhanced, enhanced, mask=mask.foreground)
    times["enhance"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    final, box = letterbox(masked, TARGET_SIZE)
    final_mask, _ = letterbox(mask.foreground, TARGET_SIZE, nearest=True)
    times["final_resize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    jpeg = encode_jpeg(final)
    times["save"] = time.perf_counter() - t0   # the file write is added by the caller

    times_ms = {k: v * 1000.0 for k, v in times.items()}
    return Result(work, mask, enhanced, final, final_mask, box, clean, restore,
                  times_ms, jpeg, raw.shape[:2])


# --------------------------------------------------------------------------
# Quality metrics (no reference image is needed)
# --------------------------------------------------------------------------
def quality_metrics(bgr: np.ndarray, region: np.ndarray) -> dict[str, float]:
    """Brightness, RMS contrast, entropy and sharpness inside `region`.

    `region` is a uint8 mask; only non-zero pixels are measured, so padding
    and masked background do not change the values.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sel = region > 0
    if not sel.any():
        return {"brightness": 0.0, "contrast": 0.0, "entropy": 0.0, "sharpness": 0.0}
    values = gray[sel]
    hist = np.bincount(values, minlength=256).astype(np.float64)
    p = hist[hist > 0] / values.size
    entropy = float(-(p * np.log2(p)).sum())
    # Erode the region so that mask and padding borders are not counted as edges.
    inner = cv2.erode(region, np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    sharpness = float(lap[inner].var()) if inner.any() else 0.0
    return {"brightness": float(values.mean()), "contrast": float(values.std()),
            "entropy": entropy, "sharpness": sharpness}


def content_region(box: tuple[int, int, int, int], size: int) -> np.ndarray:
    top, left, nh, nw = box
    region = np.zeros((size, size), np.uint8)
    region[top:top + nh, left:left + nw] = 255
    return region


def baseline_variants(work: np.ndarray, res: Result) -> dict[str, np.ndarray]:
    """Final-size images made by the baseline methods and by the pipeline."""
    clean = res.restored
    return {
        "A_resize_only": letterbox(work, TARGET_SIZE)[0],
        "B_global_HE": letterbox(global_hist_equalize(work), TARGET_SIZE)[0],
        "C_CLAHE_only": letterbox(clahe_on_lightness(work, sharpen=False), TARGET_SIZE)[0],
        "D_enhanced_no_mask": letterbox(clahe_on_lightness(clean, sharpen=True), TARGET_SIZE)[0],
        "E_full_pipeline": res.final,
    }


VARIANT_LABELS = {
    "A_resize_only": "A: resize only",
    "B_global_HE": "B: global HE",
    "C_CLAHE_only": "C: CLAHE only",
    "D_enhanced_no_mask": "D: restore + CLAHE\n+ sharpen",
    "E_full_pipeline": "E: full pipeline\n(with mask)",
}


# --------------------------------------------------------------------------
# Batch run
# --------------------------------------------------------------------------
def list_classes(root: Path) -> dict[str, list[Path]]:
    classes: dict[str, list[Path]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if files:
            classes[folder.name] = files
    return classes


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(input_dir: Path, output_dir: Path, results_dir: Path, samples_per_class: int = 3,
        make_figures: bool = True) -> None:
    classes = list_classes(input_dir)
    if not classes:
        raise SystemExit(f"No class folders with images found in {input_dir}")
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_image: list[dict] = []
    baseline_rows: list[dict] = []
    samples: dict[str, list[tuple[str, Result]]] = {c: [] for c in classes}
    skipped: list[str] = []

    # OpenCV loads code and allocates buffers on the first call. Run one image
    # first so that this start-up cost is not counted in the timing.
    process_image(next(iter(classes.values()))[0])

    wall_start = time.perf_counter()
    for cls, files in classes.items():
        out_cls = output_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        # evenly spaced sample images for the figures
        pick = set(np.linspace(0, len(files) - 1, min(samples_per_class, len(files))).astype(int).tolist())
        for i, path in enumerate(files):
            res = process_image(path)
            if res is None:
                skipped.append(str(path))
                continue
            out_path = out_cls / (path.stem + ".jpg")
            t0 = time.perf_counter()
            out_path.write_bytes(res.jpeg)
            res.times_ms["save"] += (time.perf_counter() - t0) * 1000.0

            region = content_region(res.content_box, TARGET_SIZE)
            variants = baseline_variants(res.original, res)
            before = quality_metrics(variants["A_resize_only"], region)
            fg_region = cv2.bitwise_and(region, res.final_mask)
            after = quality_metrics(res.final, fg_region)

            row = {
                "class": cls, "file": path.name,
                "raw_height": res.raw_shape[0], "raw_width": res.raw_shape[1],
                "raw_bytes": path.stat().st_size, "processed_bytes": len(res.jpeg),
                "noise_sigma": round(res.restore.noise_sigma, 3),
                "brisque_before": round(res.restore.brisque_before, 2),
                "brisque_after": round(res.restore.brisque_after, 2),
                "brisque_flagged": int(res.restore.flagged),
                "restore_filter": res.restore.filter,
                **{f"brisque_{name}": (round(res.restore.scores[name], 2) if name in res.restore.scores else "")
                   for name in RESTORE_FILTERS},
                "otsu_t": res.mask.otsu_t, "otsu_eta": round(res.mask.otsu_eta, 4),
                "dist_t": res.mask.dist_t,
                "green_fraction": round(res.mask.green_fraction, 4),
                "fg_fraction": round(res.mask.fg_fraction, 4),
                "mask_applied": int(res.mask.applied),
            }
            for stage in STAGES:
                row[f"t_{stage}_ms"] = round(res.times_ms[stage], 3)
            row["t_total_ms"] = round(sum(res.times_ms.values()), 3)
            for key, value in before.items():
                row[f"before_{key}"] = round(value, 3)
            for key, value in after.items():
                row[f"after_{key}"] = round(value, 3)
            per_image.append(row)

            for name, img in variants.items():
                reg = fg_region if name == "E_full_pipeline" else region
                m = quality_metrics(img, reg)
                baseline_rows.append({"class": cls, "file": path.name, "variant": name,
                                      "jpeg_bytes": len(encode_jpeg(img)) if name != "E_full_pipeline" else len(res.jpeg),
                                      # BRISQUE has no meaning on an image with a black masked background
                                      "brisque": round(brisque_score(img), 2) if name != "E_full_pipeline" else float("nan"),
                                      **{k: round(v, 3) for k, v in m.items()}})
            if i in pick:
                samples[cls].append((path.name, res))
        print(f"  {cls}: {len(files)} images")
    wall = time.perf_counter() - wall_start

    # ---- summaries -------------------------------------------------------
    mb = 1024.0 * 1024.0
    storage_rows = []
    for cls in list(classes) + ["TOTAL"]:
        rows = per_image if cls == "TOTAL" else [r for r in per_image if r["class"] == cls]
        raw = sum(r["raw_bytes"] for r in rows) / mb
        proc = sum(r["processed_bytes"] for r in rows) / mb
        storage_rows.append({
            "class": cls, "images": len(rows),
            "raw_MB": round(raw, 3), "processed_MB": round(proc, 3),
            "reduction_percent": round(100.0 * (1.0 - proc / raw), 2) if raw else 0.0,
            "compression_ratio": round(raw / proc, 2) if proc else 0.0,
            "mean_time_ms": round(float(np.mean([r["t_total_ms"] for r in rows])), 2),
            "mask_applied_percent": round(100.0 * float(np.mean([r["mask_applied"] for r in rows])), 1),
            "mean_fg_fraction": round(float(np.mean([r["fg_fraction"] for r in rows])), 3),
        })
    timing_rows = [{"stage": s,
                    "mean_ms": round(float(np.mean([r[f"t_{s}_ms"] for r in per_image])), 3),
                    "std_ms": round(float(np.std([r[f"t_{s}_ms"] for r in per_image])), 3)}
                   for s in STAGES]
    total_times = [r["t_total_ms"] for r in per_image]
    timing_rows.append({"stage": "TOTAL", "mean_ms": round(float(np.mean(total_times)), 3),
                        "std_ms": round(float(np.std(total_times)), 3)})

    variant_names = list(VARIANT_LABELS)
    baseline_summary = []
    for name in variant_names:
        rows = [r for r in baseline_rows if r["variant"] == name]
        baseline_summary.append({
            "variant": name,
            "mean_contrast": round(float(np.mean([r["contrast"] for r in rows])), 2),
            "mean_entropy": round(float(np.mean([r["entropy"] for r in rows])), 3),
            "mean_sharpness": round(float(np.mean([r["sharpness"] for r in rows])), 1),
            "mean_brightness": round(float(np.mean([r["brightness"] for r in rows])), 2),
            "total_MB": round(sum(r["jpeg_bytes"] for r in rows) / mb, 3),
            "mean_brisque": ("" if name == "E_full_pipeline"
                             else round(float(np.mean([r["brisque"] for r in rows])), 2)),
        })

    # quality check and restoration summary
    flagged = [r for r in per_image if r["brisque_flagged"]]
    iqa_rows = []
    for name in RESTORE_FILTERS:
        chosen = [r for r in flagged if r["restore_filter"] == name]
        tried = [r[f"brisque_{name}"] for r in flagged if r[f"brisque_{name}"] != ""]
        iqa_rows.append({
            "filter": name, "times_selected": len(chosen),
            "mean_score_all_flagged": round(float(np.mean(tried)), 2) if tried else "",
            "mean_before_when_selected": round(float(np.mean([r["brisque_before"] for r in chosen])), 2) if chosen else "",
            "mean_after_when_selected": round(float(np.mean([r["brisque_after"] for r in chosen])), 2) if chosen else "",
        })
    iqa_class_rows = []
    for cls in list(classes) + ["TOTAL"]:
        rows = per_image if cls == "TOTAL" else [r for r in per_image if r["class"] == cls]
        fl = [r for r in rows if r["brisque_flagged"]]
        iqa_class_rows.append({
            "class": cls, "images": len(rows), "flagged": len(fl),
            "mean_brisque_before": round(float(np.mean([r["brisque_before"] for r in rows])), 2),
            "mean_brisque_after": round(float(np.mean([r["brisque_after"] for r in rows])), 2),
            "flagged_mean_before": round(float(np.mean([r["brisque_before"] for r in fl])), 2) if fl else "",
            "flagged_mean_after": round(float(np.mean([r["brisque_after"] for r in fl])), 2) if fl else "",
        })
    write_csv(results_dir / "iqa_filter_summary.csv", iqa_rows)
    write_csv(results_dir / "iqa_class_summary.csv", iqa_class_rows)

    write_csv(results_dir / "metrics_per_image.csv", per_image)
    write_csv(results_dir / "storage_summary.csv", storage_rows)
    write_csv(results_dir / "timing_summary.csv", timing_rows)
    write_csv(results_dir / "baseline_comparison.csv", baseline_summary)

    lines = [f"Images processed: {len(per_image)}   skipped (not readable): {len(skipped)}",
             f"Output size: {TARGET_SIZE}x{TARGET_SIZE}, JPEG quality {JPEG_QUALITY}",
             "", "Storage (MB)",
             f"{'class':<14}{'images':>7}{'raw':>10}{'processed':>11}{'reduction':>11}{'ratio':>8}{'ms/img':>9}{'mask %':>8}"]
    for r in storage_rows:
        lines.append(f"{r['class']:<14}{r['images']:>7}{r['raw_MB']:>10.2f}{r['processed_MB']:>11.2f}"
                     f"{r['reduction_percent']:>10.2f}%{r['compression_ratio']:>8.1f}{r['mean_time_ms']:>9.1f}"
                     f"{r['mask_applied_percent']:>8.1f}")
    lines += ["", "Mean time for each stage (ms per image)"]
    lines += [f"  {r['stage']:<14}{r['mean_ms']:>9.2f}  (std {r['std_ms']:.2f})" for r in timing_rows]
    lines += [f"  Wall time for the full batch, including the baseline measurements: {wall:.1f} s",
              "", "Baseline comparison (mean over all images)",
              f"{'variant':<22}{'contrast':>10}{'entropy':>9}{'sharpness':>11}{'brightness':>12}{'total MB':>10}{'BRISQUE':>9}"]
    lines += [f"{r['variant']:<22}{r['mean_contrast']:>10.2f}{r['mean_entropy']:>9.3f}{r['mean_sharpness']:>11.1f}"
              f"{r['mean_brightness']:>12.2f}{r['total_MB']:>10.2f}{str(r['mean_brisque']) or 'n/a':>9}" for r in baseline_summary]
    lines += ["", f"Quality check (BRISQUE, lower is better; restoration if the score is above {BRISQUE_THRESHOLD:.0f})",
              f"{'class':<14}{'images':>7}{'flagged':>9}{'before':>9}{'after':>8}{'flagged before':>16}{'flagged after':>15}"]
    lines += [f"{r['class']:<14}{r['images']:>7}{r['flagged']:>9}{r['mean_brisque_before']:>9.2f}{r['mean_brisque_after']:>8.2f}"
              f"{str(r['flagged_mean_before']):>16}{str(r['flagged_mean_after']):>15}" for r in iqa_class_rows]
    lines += ["", "Filter kept for the flagged images",
              f"{'filter':<18}{'selected':>9}{'mean score on all flagged':>28}"]
    lines += [f"{r['filter']:<18}{r['times_selected']:>9}{str(r['mean_score_all_flagged']):>28}" for r in iqa_rows]
    if skipped:
        lines += ["", "Skipped files:"] + [f"  {s}" for s in skipped]
    summary = "\n".join(lines)
    (results_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)

    if make_figures:
        import figures
        figures.make_all(fig_dir, samples, per_image, storage_rows, timing_rows,
                         baseline_summary, input_dir, iqa_rows)
        print(f"\nFigures written to {fig_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant pest image preprocessing pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cv2.setRNGSeed(0)
    run(args.input, args.output, args.results, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
