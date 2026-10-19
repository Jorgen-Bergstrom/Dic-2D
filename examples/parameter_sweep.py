#!/usr/bin/env python3
"""Measure the accuracy-versus-cost trade-offs that the summary report talks about.

Runs a synthetic zero-motion pair (noise floor) and a uniform-stretch pair (bias) for a grid of
(subset, step, strain window, noise) settings, and prints a table of the measured displacement and
strain noise floors.  This is the evidence behind the guidance in `summary_report.md` (R15/§5.6)
and `DECISIONS.md` §7.

Usage
-----
    python examples/parameter_sweep.py
    python examples/parameter_sweep.py --subsets 21 31 41 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dic.config import CorrelationConfig, StrainConfig  # noqa: E402
from dic.correlation import SubsetCorrelator, make_grid  # noqa: E402
from dic.strain import compute_strains  # noqa: E402
from dic.synth import DotPattern, ground_truth_fields, render, stretch  # noqa: E402
from dic.uncertainty import compute_noise_floor  # noqa: E402

SHAPE = (512, 512)
CENTER = np.array([256.0, 256.0])


def _grid(values, xs, ys):
    nx = np.unique(xs)
    ny = np.unique(ys)
    g = np.full((ny.size, nx.size), np.nan)
    g[np.searchsorted(ny, ys), np.searchsorted(nx, xs)] = values
    return g


def measure(subset: int, step: int, window: int, noise: float, seed: int = 0) -> dict:
    """Measure the noise floor and the stretch error for one configuration."""
    pat = DotPattern.generate(SHAPE, n_dots=2400, seed=7)
    ref = render(pat, None, noise_sigma=noise, seed=1)
    cfg = CorrelationConfig(subset_size=subset, step=step, prefilter_sigma=1.0)
    xs, ys = make_grid(SHAPE, None, subset, step)
    corr = SubsetCorrelator(ref, xs, ys, cfg)

    u_rms, v_rms, s_rms, _ = compute_noise_floor(
        ref, corr, cfg, StrainConfig(window=window), noise_sigma=noise, seed=seed,
        realisations=2, max_points=900,
    )

    # stretch bias
    D = stretch(0.05, -0.02, CENTER)
    cur = render(pat, D, noise_sigma=noise, seed=2)
    p0, _ = corr.initial_guess(cur, None, None)
    res = corr.correlate_robust(cur, p0)
    m = res.valid
    gt = ground_truth_fields(D, xs, ys)
    ux = _grid(res.u, xs, ys)
    uy = _grid(res.v, xs, ys)
    valid = _grid(res.valid.astype(float), xs, ys) > 0.5
    sr = compute_strains(ux, uy, valid, float(np.median(np.diff(np.unique(xs)))),
                         float(np.median(np.diff(np.unique(ys)))), StrainConfig(window=window),
                         subset_size=subset)
    exx = sr["exx"]
    err = exx - 0.05
    exx_rms = float(np.sqrt(np.nanmean(err[np.isfinite(err)] ** 2))) if np.isfinite(err).any() else float("nan")
    return {
        "u_rms": u_rms, "v_rms": v_rms, "strain_noise": s_rms,
        "u_rms_stretch": float(np.sqrt(np.mean((res.u[m] - gt["ux"][m]) ** 2))) if m.any() else np.nan,
        "exx_rms": exx_rms,
        "coverage": float(m.mean()),
        "n_points": int(xs.size),
        "vsg": sr.gauge_length_px,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subsets", type=int, nargs="+", default=[21, 31, 41, 61])
    p.add_argument("--step", type=int, default=16)
    p.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7, 9])
    p.add_argument("--noise", type=float, default=2.0)
    args = p.parse_args(argv)

    print(f"Synthetic speckle 512x512, noise sigma = {args.noise} grey levels, "
          f"step = {args.step} px, pre-filter sigma = 1 px.\n")
    print("| subset | step | strain window | VSG [px] | points | u noise floor [px] | "
          "strain noise floor | stretch u RMS [px] | stretch exx RMS |")
    print("|---|---|---|---|---|---|---|---|---|")
    for subset in args.subsets:
        for window in args.windows:
            r = measure(subset, args.step, window, args.noise)
            print(f"| {subset} | {args.step} | {window} | {r['vsg']:.0f} | {r['n_points']} | "
                  f"{r['u_rms']:.4f} | {r['strain_noise']:.3g} | {r['u_rms_stretch']:.4f} | "
                  f"{r['exx_rms']:.3g} |")
    print("\nReading the table: the displacement noise floor falls roughly as 1/subset; the strain")
    print("noise floor falls roughly as 1/(strain window) and as 1/VSG; the spatial resolution of")
    print("the strain is the VSG.  Choose the smallest subset that keeps the displacement noise")
    print("below what you need, and the smallest strain window that keeps the strain noise below")
    print("the strain change you must resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
