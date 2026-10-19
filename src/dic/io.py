"""Image discovery, natural sorting, loading and hashing (requirements R1-R3)."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import GRAYSCALE_WEIGHTS, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

_NUM_RE = re.compile(r"(\d+)")


class ImageLoadError(RuntimeError):
    """Raised when the input image sequence cannot be loaded consistently."""


def natural_key(name: str) -> tuple:
    """Sort key that orders embedded integers numerically.

    ``image2`` precedes ``image10``; arbitrary prefixes/suffixes are handled.  Pure-text
    segments are compared case-insensitively, numbers numerically, and a text segment sorts
    before a number at the same position so that ``a`` precedes ``a1``.

    Examples
    --------
    >>> sorted(["image10.png", "image2.png"], key=natural_key)
    ['image2.png', 'image10.png']
    """
    parts = _NUM_RE.split(name)
    key: list[tuple[int, object]] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # a number
            key.append((1, int(part)))
        elif part:
            key.append((0, part.lower()))
    return tuple(key)


def discover_images(directory: str | Path) -> list[Path]:
    """Return the naturally sorted list of supported image files in *directory*.

    Parameters
    ----------
    directory : str or pathlib.Path
        Folder containing the image sequence.

    Returns
    -------
    list of pathlib.Path
        Naturally sorted image paths.

    Raises
    ------
    ImageLoadError
        If the directory is missing or contains fewer than two supported images.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ImageLoadError(f"Input image directory does not exist: {directory}")

    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda p: natural_key(p.name))
    if len(files) < 2:
        raise ImageLoadError(
            f"Need at least two images in {directory}; found {len(files)} "
            f"with extensions {SUPPORTED_EXTENSIONS}."
        )

    _warn_on_gaps(files)
    return files


def _warn_on_gaps(files: list[Path]) -> None:
    """Log a warning if the trailing integer of the file names has gaps."""
    numbers = []
    for p in files:
        m = _NUM_RE.findall(p.stem)
        if m:
            numbers.append(int(m[-1]))
    if len(numbers) < 2:
        return
    diffs = np.diff(numbers)
    expected = int(np.median(diffs)) if len(diffs) else 1
    if expected <= 0:
        expected = 1
    gaps = [
        (numbers[i], numbers[i + 1])
        for i in range(len(diffs))
        if diffs[i] != expected
    ]
    if gaps:
        shown = ", ".join(f"{a}->{b}" for a, b in gaps[:8])
        more = "" if len(gaps) <= 8 else f" (+{len(gaps) - 8} more)"
        logger.warning(
            "Image numbering is not contiguous (expected step %d): %s%s. "
            "Processing will continue in sorted order.",
            expected,
            shown,
            more,
        )


def to_grayscale(array: np.ndarray) -> np.ndarray:
    """Convert an arbitrary image array to float64 grayscale.

    * 2-D arrays are returned as-is (as ``float64``).
    * 3-D arrays are treated as ``(H, W, C)``.
    * 4-channel images are assumed RGBA and the alpha channel is dropped.
    * Colour images are converted with the Rec. 601 luma weights
      ``(0.299, 0.587, 0.114)`` -- see ``DECISIONS.md``.

    No normalisation to ``[0, 1]`` is performed: grey levels keep the native 8/16-bit scale
    so that reported noise levels are in grey levels, which is what the user sees in their
    files.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.ndim != 3:
        raise ImageLoadError(f"Unsupported image shape {array.shape}; expected 2-D or 3-D.")
    if array.shape[2] == 4:  # RGBA
        array = array[:, :, :3]
    if array.shape[2] == 1:
        return array[:, :, 0].astype(np.float64)
    if array.shape[2] < 3:
        raise ImageLoadError(f"Unsupported channel count {array.shape[2]}.")
    r, g, b = (array[:, :, i].astype(np.float64) for i in range(3))
    return GRAYSCALE_WEIGHTS[0] * r + GRAYSCALE_WEIGHTS[1] * g + GRAYSCALE_WEIGHTS[2] * b


@dataclass
class ImageSequence:
    """A loaded, validated image sequence.

    Attributes
    ----------
    paths : list of pathlib.Path
        Naturally sorted file paths.
    images : list of numpy.ndarray
        Grayscale ``float64`` images, one per path.
    shape : tuple of int
        Common ``(height, width)`` of every image.
    hashes : list of str
        SHA-256 of every input file (reproducibility, requirement 5.8).
    """

    paths: list[Path]
    images: list[np.ndarray]
    shape: tuple[int, int]
    hashes: list[str]

    def __len__(self) -> int:
        return len(self.images)


def load_sequence(
    directory: str | Path,
    *,
    max_images: int | None = None,
) -> ImageSequence:
    """Load and validate every image in *directory*.

    Parameters
    ----------
    directory : str or pathlib.Path
        Folder containing the sequence.
    max_images : int, optional
        If given, only the first *max_images* files are loaded (useful for quick tests).

    Returns
    -------
    ImageSequence

    Raises
    ------
    ImageLoadError
        If any image fails to load or the image sizes are inconsistent.
    """
    paths = discover_images(directory)
    if max_images is not None:
        paths = paths[:max_images]

    images: list[np.ndarray] = []
    hashes: list[str] = []
    shape: tuple[int, int] | None = None
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                arr = to_grayscale(np.asarray(im))
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
            raise ImageLoadError(f"Failed to load image {path}: {exc}") from exc
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ImageLoadError(
                f"Inconsistent image size: {path} is {arr.shape}, expected {shape}."
            )
        images.append(arr)
        hashes.append(_sha256(path))
        if (i + 1) % 20 == 0 or i + 1 == len(paths):
            logger.debug("Loaded %d/%d images", i + 1, len(paths))

    assert shape is not None
    logger.info(
        "Loaded %d images of %dx%d px from %s", len(images), shape[1], shape[0], directory
    )
    return ImageSequence(paths=paths, images=images, shape=shape, hashes=hashes)


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask(path: str | Path) -> np.ndarray:
    """Load a user mask (255 = valid, 0 = invalid) as a boolean array.

    Any non-zero pixel is treated as valid.  The convention matches ``Data01/masks``.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Mask file does not exist: {path}")
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > 0
