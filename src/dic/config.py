"""Central configuration and default values.

Every tunable lives here so that no numerical magic number is scattered through the code
(engineering standard 5.3).  Values are grouped by module.  Units are stated explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------------------
# Image I/O
# --------------------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".bmp",
)

#: Default grayscale conversion weights (ITU-R BT.601 / Rec. 601 luma).
GRAYSCALE_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)


# --------------------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------------------
@dataclass
class CorrelationConfig:
    """Parameters of the subset-based correlation core."""

    #: Odd subset side length in pixels.  Rule of thumb: >= 3 speckles of 3-5 px.
    subset_size: int = 41
    #: Grid step (spacing between correlation points) in pixels.
    step: int = 16
    #: B-spline interpolation order for the deformed image (3 = bicubic, 5 = biquintic).
    interp_order: int = 3
    #: B-spline interpolation order for the reference image / its gradients.
    ref_interp_order: int = 3
    #: Maximum IC-GN iterations per point.
    max_iterations: int = 30
    #: Convergence tolerance on the incremental warp parameters (px and dimensionless).
    convergence_tol: float = 1e-4
    #: Minimum zero-normalised cross-correlation for a point to be considered valid.
    min_zncc: float = 0.5
    #: Minimum sum of squared intensity gradients; guards against featureless subsets.
    min_sssig: float = 1e-3
    #: Integer-pixel search radius (px) used when the temporal initial guess is poor.
    search_radius: int = 12
    #: Correlation strategy: "auto", "temporal" or "coarse".
    init_strategy: str = "auto"
    #: If the median ZNCC of the temporal guess is below this, fall back to a coarse search.
    init_fallback_zncc: float = 0.95
    #: Coarse multi-scale search: maximum pyramid level (2**level downsample factor).
    coarse_level: int = 4
    #: Coarse multi-scale search: number of grid points per axis.
    coarse_grid: int = 9
    #: Coarse multi-scale search: integer search radius at the coarse level (px, coarse scale).
    coarse_search: int = 24
    #: Coarse search: score (median ZNCC) above which the no-rotation match is accepted.
    coarse_good_score: float = 0.8
    #: Coarse search: rotation angle step for the candidate search (deg; 0 disables).
    coarse_angle_step: float = 10.0
    #: Coarse search: maximum candidate rotation magnitude (deg; 0 disables).
    coarse_angle_max: float = 45.0
    #: Reference strategy: "first" (always the original reference) or "incremental".
    reference: str = "first"
    #: Incremental mode: promote the reference every N frames (1 = every frame).
    update_interval: int = 1
    #: For incremental mode, update the reference when median ZNCC drops below this.
    update_zncc: float = 0.85
    #: Gaussian pre-filter sigma (px) applied to both images before correlation (0 = off).
    prefilter_sigma: float = 1.0


# --------------------------------------------------------------------------------------
# ROI / masking
# --------------------------------------------------------------------------------------
@dataclass
class MaskConfig:
    """Parameters of the automatic region-of-interest detector."""

    #: Mask source: "auto", "user" or "none".
    mode: str = "auto"
    #: Window (px, odd) for the local variance / gradient-energy map.
    window: int = 21
    #: Threshold as a fraction between the background level and the speckle level
    #: (0 = keep everything above the background, 1 = keep only the brightest texture).
    threshold_fraction: float = 0.25
    #: Morphological closing/opening structuring-element radius (px).
    morphology_radius: int = 3
    #: Drop connected components smaller than this fraction of the largest component.
    min_component_fraction: float = 0.02
    #: Absolute minimum component area in pixels.
    min_component_area: int = 512
    #: Erode the mask by (subset_size // 2).  The correlation grid *always* requires the full
    #: subset footprint to be inside the ROI (see ``correlation.make_grid``), so this only
    #: affects the saved/displayed mask.  Off by default so the saved mask is the true ROI.
    erode_for_subset: bool = False


# --------------------------------------------------------------------------------------
# Strain
# --------------------------------------------------------------------------------------
@dataclass
class StrainConfig:
    """Parameters of the virtual-strain-gauge (plane-fit) strain estimator."""

    #: Number of displacement samples per axis in the fitting window (odd, >= 3).
    window: int = 5
    #: Warn when |engineering strain| exceeds this (small-strain validity limit).
    engineering_validity_limit: float = 0.05


# --------------------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------------------
@dataclass
class UncertaintyConfig:
    """Parameters of the uncertainty / noise-floor machinery."""

    #: Estimate per-point sigma from the correlation Hessian and residual.
    per_point: bool = True
    #: Compute an empirical zero-motion noise floor.
    noise_floor: bool = True
    #: Number of synthetic zero-motion realisations used for the noise floor.
    noise_floor_realisations: int = 3
    #: If > 0, override the image-noise estimate with this grey-level sigma.
    image_noise_sigma: float = 0.0


# --------------------------------------------------------------------------------------
# Output / visualization
# --------------------------------------------------------------------------------------
@dataclass
class OutputConfig:
    """Parameters controlling figures, animations and machine-readable output."""

    #: Render the per-frame displacement and strain overlays.
    figures: bool = True
    #: Render the extra visualizations (curves, profiles, animations, quiver).
    extras: bool = True
    #: Figure resolution (dots per inch).
    dpi: int = 110
    #: Contour levels for the overlay plots.
    contour_levels: int = 15
    #: Opacity of the field overlay on the base image (0-1).
    overlay_alpha: float = 0.65
    #: Which frame the field is drawn on: "deformed" (the current image, with the field placed
    #: at the material positions X+u, so the specimen visibly deforms) or "reference" (the
    #: undeformed image with the field at the reference positions).
    overlay_frame: str = "deformed"
    #: Animation frames per second.
    animation_fps: int = 12
    #: Animation file format: "mp4" (needs ffmpeg) or "gif".
    animation_format: str = "gif"
    #: Write per-frame NPZ field archives.
    save_npz: bool = True
    #: Write the long-form per-point CSV.
    save_csv: bool = False


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Top-level run configuration."""

    #: Seconds between consecutive frames (for the time axis).  <=0 means "frame index".
    dt: float = 1.0
    #: Physical length of one pixel in user units (e.g. mm/px).
    scale: float = 1.0
    #: Unit label for the physical scale.
    unit: str = "px"
    #: Coverage (valid fraction) below which the run fails loudly.
    min_coverage: float = 0.30
    #: Worker processes for figure rendering (1 = serial).
    jobs: int = 1
    #: Master random seed for any stochastic step (reproducibility).
    seed: int = 0
    #: Log level name.
    log_level: str = "INFO"

    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    strain: StrainConfig = field(default_factory=StrainConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dictionary of every setting."""
        return asdict(self)
