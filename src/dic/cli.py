"""Command-line interface.

``python -m dic IMAGES OUTPUT [options]`` runs the analysis;
``python -m dic --validate OUTPUT`` runs the synthetic validation suite.

Every option documents its default and its unit in ``--help``.  Exit codes:

====  ==========================================================
Code  Meaning
====  ==========================================================
0     success
1     unexpected error
2     command-line usage error (argparse)
3     input problem (missing directory, unreadable images, <2 frames)
4     coverage collapsed below ``--min-coverage`` (unless lowered)
5     one or more validation cases failed
====  ==========================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import (
    CorrelationConfig,
    MaskConfig,
    OutputConfig,
    RunConfig,
    StrainConfig,
    UncertaintyConfig,
)
from .io import ImageLoadError
from .outputs import write_all
from .pipeline import run_pipeline
from .progress import ProgressReporter
from .report import write_summary

logger = logging.getLogger("dic")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 3
EXIT_COVERAGE = 4
EXIT_VALIDATION = 5


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (kept public so tests can introspect it)."""
    p = argparse.ArgumentParser(
        prog="dic",
        description=(
            "2D Digital Image Correlation (DIC) on a sequence of speckle images. "
            "Computes full-field displacement and strain with built-in uncertainty "
            "quantification and a synthetic validation suite."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Coordinate and unit conventions: images are indexed [row=y, col=x]; x points right, "
            "y points down; displacements are in pixels unless --scale is given (then in the "
            "chosen --unit); strains are dimensionless. The first image in natural sort order is "
            "the undeformed reference. Engineering strain is only valid for small strain; use the "
            "Green-Lagrange fields at large deformation."
        ),
    )
    p.add_argument("images", nargs="?", help="Directory of speckle images (read-only input).")
    p.add_argument("output", help="Directory to populate with results (created if missing).")

    g = p.add_argument_group("Acquisition")
    g.add_argument("--dt", type=float, default=1.0,
                   help="Seconds between frames; <=0 uses the frame index. [s]")
    g.add_argument("--scale", type=float, default=1.0,
                   help="Physical length of one pixel. [unit/px]")
    g.add_argument("--unit", type=str, default="px",
                   help="Unit label for --scale (e.g. mm, um, in).")
    g.add_argument("--max-images", type=int, default=None,
                   help="Process only the first N images (debugging).")

    g = p.add_argument_group("Correlation")
    g.add_argument("--subset", type=int, default=41,
                   help="Odd subset side length; >= 3 speckles of 3-5 px. [px]")
    g.add_argument("--step", type=int, default=16,
                   help="Grid spacing between correlation points. [px]")
    g.add_argument("--interp-order", type=int, choices=(1, 3, 5), default=3,
                   help="B-spline order for the deformed image (3=bicubic, 5=biquintic).")
    g.add_argument("--prefilter", type=float, default=1.0,
                   help="Gaussian pre-filter sigma applied to both images; 0 disables. [px]")
    g.add_argument("--reference", choices=("first", "incremental"), default="first",
                   help="'first': always correlate to the original reference (no accumulation). "
                        "'incremental': update the reference for large deformation "
                        "(accumulates error).")
    g.add_argument("--update-interval", type=int, default=1,
                   help="Incremental mode: promote the reference every N frames.")
    g.add_argument("--init", choices=("auto", "temporal", "coarse"), default="auto",
                   help="Initial-guess strategy.")
    g.add_argument("--search-radius", type=int, default=12,
                   help="Integer search radius for large inter-frame motion. [px]")
    g.add_argument("--max-iter", type=int, default=30,
                   help="Maximum IC-GN iterations per point.")
    g.add_argument("--conv-tol", type=float, default=1e-4,
                   help="IC-GN convergence tolerance on the incremental warp. [px, -]")
    g.add_argument("--min-zncc", type=float, default=0.5,
                   help="Minimum ZNCC for a point to be valid. [-]")
    g.add_argument("--min-coverage", type=float, default=0.30,
                   help="Fail if the valid-point fraction drops below this. [-]")

    g = p.add_argument_group("Region of interest / masking")
    g.add_argument("--mask", type=str, default=None,
                   help="User mask image (255=valid) or directory of per-frame masks.")
    g.add_argument("--mask-mode", choices=("auto", "user", "none"), default="auto",
                   help="'auto' detect the speckled region, 'user' use --mask, 'none' use all.")
    g.add_argument("--mask-index", type=int, default=0,
                   help="Which frame's mask to use when --mask is a directory.")
    g.add_argument("--mask-window", type=int, default=21,
                   help="Local-variance window for automatic masking. [px]")
    g.add_argument("--mask-threshold", type=float, default=0.25,
                   help="Auto-mask threshold as a fraction of the texture dynamic range. [-]")

    g = p.add_argument_group("Strain")
    g.add_argument("--strain-window", type=int, default=5,
                   help="Odd number of displacement samples per axis in the plane fit. [points]")

    g = p.add_argument_group("Uncertainty")
    g.add_argument("--noise-sigma", type=float, default=0.0,
                   help="Override the estimated image noise. [grey levels]")
    g.add_argument("--no-noise-floor", action="store_true",
                   help="Skip the empirical zero-motion noise-floor measurement.")

    g = p.add_argument_group("Output")
    g.add_argument("--no-figures", action="store_true", help="Do not write per-frame figures.")
    g.add_argument("--no-extras", action="store_true",
                   help="Skip curves, profiles, uncertainty map and animations.")
    g.add_argument("--dpi", type=int, default=110, help="Figure resolution. [dpi]")
    g.add_argument("--animation-format", choices=("gif", "mp4"), default="gif",
                   help="Animation container (mp4 needs ffmpeg).")
    g.add_argument("--overlay", choices=("deformed", "reference"), default="deformed",
                   dest="overlay_frame",
                   help="Draw the field on the current image at the material positions "
                        "('deformed', so the specimen visibly deforms) or on the undeformed "
                        "reference image at the reference positions ('reference').")
    g.add_argument("--csv", action="store_true",
                   help="Also write the long-form per-point CSV (large).")
    g.add_argument("--jobs", type=int, default=1,
                   help="Worker processes for figure rendering.")

    g = p.add_argument_group("Runtime")
    g.add_argument("--seed", type=int, default=0, help="Master random seed.")
    g.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity.")
    g.add_argument("--log-file", default=None,
                   help="Log file path (default: <output>/run.log).")
    g.add_argument("--quiet", action="store_true", help="Do not print the progress bar.")

    g = p.add_argument_group("Validation")
    g.add_argument("--validate", action="store_true",
                   help="Run the synthetic validation suite instead of a user analysis.")
    g.add_argument("--validate-quick", action="store_true",
                   help="Use a fast configuration for the validation suite.")
    g.add_argument("--version", action="version", version=f"dic {__version__}")
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Translate parsed arguments into a :class:`RunConfig`."""
    cfg = RunConfig(
        dt=args.dt,
        scale=args.scale,
        unit=args.unit,
        min_coverage=args.min_coverage,
        jobs=args.jobs,
        seed=args.seed,
        log_level=args.log_level,
    )
    cfg.correlation = CorrelationConfig(
        subset_size=args.subset,
        step=args.step,
        interp_order=args.interp_order,
        ref_interp_order=args.interp_order,
        max_iterations=args.max_iter,
        convergence_tol=args.conv_tol,
        min_zncc=args.min_zncc,
        search_radius=args.search_radius,
        init_strategy=args.init,
        reference=args.reference,
        update_interval=args.update_interval,
        prefilter_sigma=args.prefilter,
    )
    cfg.mask = MaskConfig(
        mode=args.mask_mode,
        window=args.mask_window,
        threshold_fraction=args.mask_threshold,
    )
    cfg.strain = StrainConfig(window=args.strain_window)
    cfg.uncertainty = UncertaintyConfig(
        per_point=True,
        noise_floor=not args.no_noise_floor,
        image_noise_sigma=args.noise_sigma,
    )
    cfg.output = OutputConfig(
        figures=not args.no_figures,
        extras=not args.no_extras,
        dpi=args.dpi,
        animation_format=args.animation_format,
        overlay_frame=args.overlay_frame,
        save_csv=args.csv,
    )
    return cfg


def _setup_logging(level: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    root.addHandler(sh)


def _run_analysis(args: argparse.Namespace) -> int:
    if not args.images:
        logger.error("An input image directory is required (or use --validate).")
        return EXIT_INPUT
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output / "run.log"
    _setup_logging(args.log_level, log_file)
    logger.info("dic %s - 2D DIC analysis", __version__)
    cfg = config_from_args(args)

    reporter = None if args.quiet else ProgressReporter(0, "correlate")
    try:
        result = run_pipeline(
            args.images,
            cfg,
            user_mask_path=args.mask,
            user_mask_index=args.mask_index,
            max_images=args.max_images,
            progress=reporter,
        )
    except ImageLoadError as exc:
        logger.error("%s", exc)
        return EXIT_INPUT
    except (RuntimeError, ValueError) as exc:
        logger.error("Analysis failed: %s", exc)
        return EXIT_ERROR

    logger.info(
        "Correlation done: %d frames, %d points, %.1f s (%.2f s/frame)",
        result.n_frames,
        result.xs.size,
        result.timings["correlation_s"],
        result.timings["correlation_s"] / max(result.n_frames, 1),
    )

    write_all(result, output)
    write_summary(result, output)

    if cfg.output.figures or cfg.output.extras:
        from .visualization import render_all_figures

        fig_reporter = None if args.quiet else ProgressReporter(result.n_frames, "figures")
        render_all_figures(result, output, progress=fig_reporter)
        if fig_reporter:
            fig_reporter.close()

    covs = [f.coverage for f in result.frames]
    if covs and min(covs) < cfg.min_coverage:
        logger.error(
            "Coverage collapsed to %.1f%% (below --min-coverage %.0f%%). "
            "Results were written; lower --min-coverage to accept them.",
            100 * min(covs),
            100 * cfg.min_coverage,
        )
        return EXIT_COVERAGE

    logger.info("Done. Results in %s", output)
    return EXIT_OK


def _run_validation(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output / "validation.log"
    _setup_logging(args.log_level, log_file)
    from .validation import run_validation

    reporter = None if args.quiet else ProgressReporter(0, "validate")
    dataset_dir = args.images if args.images else None
    path, cases = run_validation(
        output, quick=args.validate_quick, dataset_dir=dataset_dir, progress=reporter
    )
    n_fail = sum(not c.passed for c in cases)
    logger.info("Validation report: %s", path)
    if n_fail:
        logger.error("%d/%d validation cases FAILED.", n_fail, len(cases))
        return EXIT_VALIDATION
    logger.info("All %d validation cases passed.", len(cases))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate:
            return _run_validation(args)
        return _run_analysis(args)
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("Interrupted.")
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 - final safety net
        logging.getLogger("dic").exception("Unexpected error")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
