"""Human-readable summary report (R13) and the user-facing accuracy interpretation (5.6)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .pipeline import RunResult

logger = logging.getLogger(__name__)


def _fmt(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}g}"


def build_summary(result: RunResult) -> str:
    """Build the Markdown summary report text."""
    cfg = result.cfg
    corr = cfg.correlation
    unc = result.uncertainty
    lines: list[str] = []
    add = lines.append

    add("# 2D DIC analysis - summary report\n")
    add(f"* Input directory: `{result.image_paths[0].parent}`")
    add(f"* Frames analysed: **{result.n_frames}** ({result.shape[1]} x {result.shape[0]} px)")
    add(f"* Reference frame: `{result.image_paths[0].name}` (first in natural order)")
    add(f"* Time step: {cfg.dt} s/frame; physical scale {cfg.scale} {cfg.unit}/px")
    add(f"* Correlation grid: {result.nx} x {result.ny} = {result.xs.size} points, "
        f"step {result.dx:.0f} x {result.dy:.0f} px")
    add(f"* ROI: {result.mask_source}; coverage {100 * result.mask.mean():.1f}% of the image")
    add("")

    add("## Settings actually used\n")
    add("| Setting | Value |")
    add("|---|---|")
    add("| Correlation criterion | ZNSSD |")
    add("| Shape function | affine (6-parameter) |")
    add(f"| Optimiser | inverse-compositional Gauss-Newton, tol {corr.convergence_tol:g}, "
        f"max {corr.max_iterations} iters |")
    add(f"| Sub-pixel interpolant | B-spline order {corr.interp_order} "
        f"(reference order {corr.ref_interp_order}) |")
    add(f"| Pre-filter | Gaussian sigma {corr.prefilter_sigma:g} px |")
    add(f"| Subset size | {corr.subset_size} px |")
    add(f"| Grid step | {corr.step} px |")
    add(f"| Reference strategy | {corr.reference} |")
    add(f"| Min ZNCC | {corr.min_zncc} |")
    add(f"| Strain window | {cfg.strain.window} x {cfg.strain.window} points |")
    if result.frames:
        add(f"| Virtual strain gauge | {result.frames[-1].strain.gauge_length_px:.0f} px |")
    add("")

    # ---- coverage & quality ---------------------------------------------------------
    add("## Per-frame coverage and quality\n")
    add("| Frame | Image | t | Coverage | ZNCC med | mean sigma_u | mean sigma_v | mean Eeng_xx | max |Eeng_xx| |")
    add("|---|---|---|---|---|---|---|---|---|")
    step = max(1, result.n_frames // 40)
    shown = list(range(0, result.n_frames, step))
    if shown[-1] != result.n_frames - 1:
        shown.append(result.n_frames - 1)
    for i in shown:
        f = result.frames[i]
        Eeng_xx = f.strain["Eeng_xx"]
        max_Eeng_xx = np.nanmax(np.abs(Eeng_xx)) if np.isfinite(Eeng_xx).any() else np.nan
        add(
            f"| {f.index + 1} | {f.name} | {f.t_load:.3f} | {100 * f.coverage:.1f}% | "
            f"{_fmt(np.nanmedian(f.zncc), 5)} | {_fmt(f.mean_sigma_u)} | {_fmt(f.mean_sigma_v)} | "
            f"{_fmt(np.nanmean(Eeng_xx))} | {_fmt(max_Eeng_xx)} |"
        )
    add("")
    covs = np.array([f.coverage for f in result.frames])
    add(f"* Mean coverage: **{100 * covs.mean():.1f}%**; minimum "
        f"**{100 * covs.min():.1f}%** at frame {int(np.argmin(covs)) + 1}.")
    add("* All non-converged, out-of-frame or low-correlation points are written as `NaN` and "
        "excluded from every statistic. Nothing is fabricated.")
    add("")

    # ---- warnings --------------------------------------------------------------------
    if result.warnings:
        add("## Warnings\n")
        for w in result.warnings:
            add(f"* :warning: {w}")
        add("")

    # ---- uncertainty -----------------------------------------------------------------
    add("## Accuracy and uncertainty\n")
    if unc is not None:
        add(f"* Estimated image noise: **{_fmt(unc.image_noise_sigma, 3)} grey levels** "
            f"({unc.image_noise_method}).")
        add(f"* **Noise floor** (zero true motion, same speckle/settings/noise): "
            f"u RMS **{_fmt(unc.noise_floor_u_rms_px)} px**, "
            f"v RMS **{_fmt(unc.noise_floor_v_rms_px)} px**, "
            f"strain RMS **{_fmt(unc.noise_floor_strain_rms)}** "
            f"({unc.noise_floor_points} points).")
        add(f"* Mean per-point model uncertainty over the run: "
            f"sigma_u {_fmt(unc.mean_sigma_u_px)} px, sigma_v {_fmt(unc.mean_sigma_v_px)} px.")
        add("")
        add("### How to read these numbers\n")
        add("* The **noise floor** is the displacement/strain the method produces when the true "
            "motion is zero.  Any feature in your data smaller than roughly 2-3x the noise floor "
            "is not resolvable.")
        add("* The **per-point sigma_u/sigma_v** (written into every `fields/*.npz`) is a "
            "model-based estimate `sigma_r * sqrt(diag(H^-1))` from the matching residual and the "
            "correlation Hessian.  It varies across the field with local contrast; the maps are "
            "in `extras/uncertainty.png`.")
        add("* Displacement is reported relative to the first image; strains are averages over the "
            "virtual strain gauge, so the strain field is smoother than the displacement field.")
        add("")
        add(_interpret(cfg, unc))
    else:
        add("* Uncertainty analysis was disabled for this run.")
    add("")

    add("## Files written\n")
    add("* `params.json` - full settings, package versions, input file names + SHA-256")
    add("* `mask/mask_used.png`, `mask/mask_used.npz` - the ROI actually used")
    add("* `grid.csv` - correlation grid coordinates")
    add("* `fields/frame_XXXX.npz` - per-frame displacement, quality and strain arrays")
    add("* `displacement_timeseries.csv`, `strain_timeseries.csv` - per-frame summaries")
    add("* `figures/displacement/`, `figures/strain/` - one overlay per input image")
    add("* `extras/` - curves, line profiles, uncertainty map; `animations/` - field animations")
    add("")
    return "\n".join(lines)


def _interpret(cfg, unc) -> str:
    """Turn the measured numbers into user-facing guidance (requirement 5.6)."""
    nf_u = unc.noise_floor_u_rms_px
    nf_e = unc.noise_floor_strain_rms
    subset = cfg.correlation.subset_size
    window = cfg.strain.window
    step = cfg.correlation.step
    vsg = (window - 1) * step + subset
    lines = ["### What this means for your data\n"]
    if np.isfinite(nf_u):
        lines.append(
            f"* With **subset {subset} px, step {step} px, strain window {window}x{window}, "
            f"image noise {_fmt(unc.image_noise_sigma, 3)} grey levels**, expect a displacement "
            f"noise floor of about **{_fmt(nf_u, 3)} px** "
            f"({_fmt(nf_u * cfg.scale, 3)} {cfg.unit})."
        )
    if np.isfinite(nf_e):
        lines.append(
            f"* Expect a strain noise floor of about **{_fmt(nf_e, 3)}** "
            f"({_fmt(100 * nf_e, 3)}%).  This is the **smallest strain change your data can "
            f"resolve**: differences below it are noise."
        )
    lines.append(
        f"* The strain is averaged over a **virtual strain gauge of ~{vsg:.0f} px** "
        f"(`(window-1)*step + subset`).  A strain peak narrower than that is attenuated.  "
        f"Halving the window halves the gauge length but roughly doubles the strain noise; "
        f"doubling it does the opposite."
    )
    lines.append(
        "* Displacement noise scales roughly linearly with image noise and inversely with the "
        "speckle contrast; strain noise scales with displacement noise divided by the gauge "
        "length.  If you need lower noise, enlarge the subset (loses spatial resolution near "
        "edges) and/or the strain window."
    )
    lines.append(
        "* **2D-DIC caveats:** out-of-plane motion, lens distortion, a non-parallel camera axis "
        "and a non-planar specimen all produce bias that no amount of tuning removes.  The "
        "reported nominal strain (V - I, Abaqus NE) is valid at finite strain; true "
        "(logarithmic/Hencky) strain T = ln(V) is also written for comparison."
    )
    return "\n".join(lines)


def write_summary(result: RunResult, output_dir: str | Path) -> Path:
    """Write ``summary_report.md``."""
    output_dir = Path(output_dir)
    path = output_dir / "summary_report.md"
    path.write_text(build_summary(result))
    logger.info("Wrote summary report to %s", path)
    return path
