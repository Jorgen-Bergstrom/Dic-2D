"""Region-of-interest detection and masking (requirement R5).

The automatic detector distinguishes *speckled* (textured) from *flat* (background) regions using
the local grey-level standard deviation, thresholds it with Otsu's method (or a fixed fraction of
the texture dynamic range when Otsu is not meaningful), cleans the result with morphology and
connected-component filtering, and finally erodes it so that every correlation subset lies
entirely inside the specimen.

Assumptions
-----------
* The specimen carries a speckle (or otherwise textured) pattern; the background is flat or at
  least has much lower local texture than the specimen.
* The illumination is roughly uniform (a strong vignette can make part of a flat background look
  textured, or part of the specimen look flat).

Failure modes
-------------
* **A textured but non-specimen background** (a printed grid, a fixture with a pattern) is kept:
  the detector separates *textured from flat*, not *specimen from background*.
* **A smooth specimen region** (very low local variance, e.g. a polished or over-exposed area)
  is discarded, and the corresponding correlation points are lost.
* **A frame with no flat background at all** is handled explicitly: a unimodal local-texture
  distribution is detected and the whole frame is kept.
* **Strong illumination gradients** can bias the local-variance map; raise ``--mask-window`` or
  supply ``--mask`` in that case.

A user mask (``--mask``, 255 = valid) always overrides the automatic detection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import MaskConfig
from .io import load_mask

logger = logging.getLogger(__name__)


def local_std(image: np.ndarray, window: int) -> np.ndarray:
    """Local standard deviation of *image* over a *window* x *window* box."""
    img = np.asarray(image, dtype=np.float64)
    size = (window, window)
    mean = ndimage.uniform_filter(img, size=size, mode="mirror")
    mean_sq = ndimage.uniform_filter(img * img, size=size, mode="mirror")
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(var)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold on a 1-D array of values.

    Parameters
    ----------
    values : numpy.ndarray
        Flattened sample values.
    bins : int
        Histogram bins.

    Returns
    -------
    float
        The threshold that maximises the between-class variance.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=bins)
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total <= 0:
        return float(np.mean(v))
    p = hist / total
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    valid = (w0 > 0) & (w1 > 0)
    between = np.zeros_like(w0)
    between[valid] = (mu_t * w0[valid] - mu[valid]) ** 2 / (w0[valid] * w1[valid])
    return float(centers[int(np.argmax(between))])


def _disk(radius: int) -> np.ndarray:
    """Boolean disk structuring element."""
    r = max(0, int(radius))
    if r == 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def auto_mask(image: np.ndarray, cfg: MaskConfig, subset_size: int) -> np.ndarray:
    """Detect the speckled region of *image* automatically.

    Parameters
    ----------
    image : numpy.ndarray
        Reference image.
    cfg : MaskConfig
        Detector parameters.
    subset_size : int
        Correlation subset size; the final mask is eroded by ``subset_size // 2`` so that every
        subset footprint is inside the ROI.

    Returns
    -------
    numpy.ndarray
        Boolean mask (``True`` = valid).
    """
    std = local_std(image, cfg.window)
    lo, hi = float(np.percentile(std, 5)), float(np.percentile(std, 99))
    if lo > 0.5 * hi or hi <= 0:
        # unimodal texture: there is no flat background to remove
        logger.info(
            "Auto-mask: unimodal local texture (floor=%.2f, peak=%.2f) - the frame appears "
            "fully speckled; using the whole image as the ROI.",
            lo,
            hi,
        )
        return np.ones_like(std, dtype=bool)
    frac_thresh = lo + cfg.threshold_fraction * max(hi - lo, 0.0)
    otsu = otsu_threshold(std)
    # Otsu is the textbook choice for a bimodal (speckle vs flat) distribution, but it is
    # meaningless when the whole frame is speckled.  Use it only when it produces a plausible
    # split; otherwise fall back to a fixed fraction of the texture dynamic range.
    thresh = frac_thresh
    method = "fraction"
    if lo < otsu < hi:
        cov = float(np.mean(std > otsu))
        if 0.02 <= cov <= 0.90:
            thresh = otsu
            method = "Otsu"
    mask = std > thresh
    if mask.mean() > 0.97:
        logger.info(
            "Auto-mask: the frame appears fully speckled (texture fraction %.1f%%); "
            "using the whole image as the ROI.",
            100 * mask.mean(),
        )
        mask = np.ones_like(mask)
    else:
        logger.info(
            "Auto-mask: %s threshold=%.3f (Otsu=%.3f, floor=%.3f, peak=%.3f), "
            "initial coverage=%.1f%%",
            method,
            thresh,
            otsu,
            lo,
            hi,
            100.0 * mask.mean(),
        )

    disk = _disk(cfg.morphology_radius)
    mask = ndimage.binary_closing(mask, structure=disk, border_value=0)
    mask = ndimage.binary_opening(mask, structure=disk, border_value=0)
    mask = ndimage.binary_fill_holes(mask)

    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
        largest = float(sizes.max())
        keep = np.zeros(n + 1, dtype=bool)
        for i, s in enumerate(sizes, start=1):
            if s >= cfg.min_component_area and s >= cfg.min_component_fraction * largest:
                keep[i] = True
        mask = keep[labeled]
        logger.info("Auto-mask: kept %d of %d connected components.", int(keep.sum()), n)

    if cfg.erode_for_subset:
        r = subset_size // 2
        mask = ndimage.binary_erosion(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return mask


def resolve_user_mask(path: str | Path, shape: tuple[int, int], index: int = 0) -> np.ndarray:
    """Load a user mask, either a single file or a directory of per-frame masks.

    Parameters
    ----------
    path : str or pathlib.Path
        Mask file, or a directory containing ``maskNNNN.png`` style files.
    shape : tuple of int
        Expected ``(height, width)``.
    index : int
        Which frame's mask to use when *path* is a directory (default: the reference frame).

    Returns
    -------
    numpy.ndarray
        Boolean mask.
    """
    from .io import natural_key

    path = Path(path)
    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff", ".bmp")),
            key=lambda p: natural_key(p.name),
        )
        if not files:
            raise FileNotFoundError(f"No mask images found in {path}")
        index = int(np.clip(index, 0, len(files) - 1))
        logger.info("User mask: using %s from %s", files[index].name, path)
        chosen = files[index]
    else:
        chosen = path
    mask = load_mask(chosen)
    if mask.shape != shape:
        raise ValueError(
            f"User mask {chosen} has shape {mask.shape}, expected {shape}."
        )
    return mask


def build_mask(
    reference: np.ndarray,
    cfg: MaskConfig,
    subset_size: int,
    *,
    user_mask_path: str | Path | None = None,
    user_mask_index: int = 0,
) -> tuple[np.ndarray, str]:
    """Resolve the ROI mask according to ``cfg.mode``.

    Returns
    -------
    mask : numpy.ndarray
        Boolean valid mask.
    source : str
        Human-readable description of where the mask came from.
    """
    mode = cfg.mode
    if mode == "none":
        return np.ones(reference.shape, dtype=bool), "none (whole image)"
    if mode == "user" or (mode == "auto" and user_mask_path is not None):
        if user_mask_path is None:
            raise ValueError("--mask-mode user requires --mask PATH.")
        mask = resolve_user_mask(user_mask_path, reference.shape, user_mask_index)
        if cfg.erode_for_subset:
            r = subset_size // 2
            mask = ndimage.binary_erosion(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        return mask, f"user mask {user_mask_path}"
    mask = auto_mask(reference, cfg, subset_size)
    if not mask.any():
        logger.warning("Automatic masking produced an empty ROI; falling back to the whole image.")
        mask = np.ones(reference.shape, dtype=bool)
        return mask, "auto (empty -> whole image)"
    if mask.all():
        return mask, "auto (fully speckled frame -> whole image)"
    return mask, "auto (local-variance detection)"
