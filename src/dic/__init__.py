"""dic - a research-grade 2D Digital Image Correlation package.

Public API
----------
``RunConfig`` and the configuration dataclasses, the correlation core, strain estimation,
masking, uncertainty, visualization, validation and the CLI entry point.

The package is usable both as a library (``from dic import RunConfig, run_pipeline``) and as a
command-line application (``python -m dic --help``).
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    CorrelationConfig,
    MaskConfig,
    OutputConfig,
    RunConfig,
    StrainConfig,
    UncertaintyConfig,
)

__all__ = [
    "__version__",
    "RunConfig",
    "CorrelationConfig",
    "MaskConfig",
    "StrainConfig",
    "UncertaintyConfig",
    "OutputConfig",
]
