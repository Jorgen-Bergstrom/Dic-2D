"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dic.synth import DotPattern, render  # noqa: E402


@pytest.fixture(scope="session")
def speckle_pattern() -> DotPattern:
    """A small, fixed speckle pattern for fast tests."""
    return DotPattern.generate((220, 220), n_dots=420, seed=3, softness=0.9)


@pytest.fixture(scope="session")
def speckle_image(speckle_pattern: DotPattern) -> np.ndarray:
    """Noiseless reference image of the fixture pattern."""
    return render(speckle_pattern, None, noise_sigma=0.0)


@pytest.fixture
def tmp_images(tmp_path: Path):
    """Write a list of arrays as a PNG sequence and return the directory."""

    def _write(images, prefix: str = "image", ext: str = ".png") -> Path:
        from PIL import Image

        d = tmp_path / "seq"
        d.mkdir(exist_ok=True)
        for i, img in enumerate(images):
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                d / f"{prefix}{i + 1:04d}{ext}"
            )
        return d

    return _write
