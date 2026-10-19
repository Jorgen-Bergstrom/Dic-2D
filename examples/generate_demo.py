#!/usr/bin/env python3
"""Generate a demo DIC dataset with known ground truth.

The provided ``Data01/`` dataset only exercises a large uniform stretch.  This generator creates
sequences that exercise the modes it does **not** cover:

* rigid rotation and translation (including a large single-step rotation),
* empty background regions that require automatic ROI detection,
* a linear illumination change,
* higher image noise.

Every sequence is rendered analytically (no interpolation of a raster), so the ground truth is
exact.  The deformation parameters and the analytic fields on the correlation grid are written
next to the images so the user can check any result themselves.

Usage
-----
    python examples/generate_demo.py --output examples/demo_data
    python examples/generate_demo.py --preset masked --output /tmp/masked
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dic.synth import (  # noqa: E402
    AffineDeformation,
    Deformation,
    DotPattern,
    NonUniformDeformation,
    ground_truth_fields,
    render,
    rotation,
    stretch,
)

SIZE = 640
CENTER = np.array([SIZE / 2.0, SIZE / 2.0])


def _rotation_sequence(n: int) -> tuple[list[Deformation], str]:
    """1 deg/frame to 30 deg plus a growing translation."""
    out = []
    for i in range(n):
        t = i / (n - 1)
        D = rotation(np.radians(30.0 * t), CENTER)
        out.append(
            AffineDeformation(D.F, np.array([25.0 * t, -15.0 * t]), CENTER)
        )
    return out, "rigid rotation 0->30 deg with translation 0->(25,-15) px"


def _masked_sequence(n: int) -> tuple[list[Deformation], str]:
    """Uniform stretch of a speckled rectangle on an empty background."""
    out = [stretch(0.08 * i / (n - 1), -0.03 * i / (n - 1), CENTER) for i in range(n)]
    return out, "uniform stretch 0->8% of a central speckled patch (flat background elsewhere)"


def _illumination_sequence(n: int) -> tuple[list[Deformation], str]:
    """Sub-pixel translation with a linear illumination change."""
    out = [AffineDeformation(np.eye(2), np.array([3.0 * i / (n - 1), 2.0 * i / (n - 1)]), CENTER)
           for i in range(n)]
    return out, "translation 0->(3,2) px with gain 1.0->1.25 and a linear illumination ramp"


def _noisy_sequence(n: int) -> tuple[list[Deformation], str]:
    """Uniform stretch with high noise (2% of the dynamic range)."""
    out = [stretch(0.15 * i / (n - 1), -0.05 * i / (n - 1), CENTER) for i in range(n)]
    return out, "uniform stretch 0->15% with sigma = 5.1 grey levels (2% of 255)"


def _nonuniform_sequence(n: int) -> tuple[list[Deformation], str]:
    """A smooth non-uniform field."""
    out = [
        NonUniformDeformation(
            a=6.0 * i / (n - 1), b=4.0 * i / (n - 1),
            kx=2 * np.pi / 380.0, ky=2 * np.pi / 460.0,
            shear=(0.03 * i / (n - 1), 0.01 * i / (n - 1),
                   -0.01 * i / (n - 1), -0.02 * i / (n - 1)),
            center=CENTER,
        )
        for i in range(n)
    ]
    return out, "smooth sinusoidal + linear non-uniform displacement field"


PRESETS = {
    "rotation": (_rotation_sequence, dict(noise=2.0)),
    "masked": (_masked_sequence, dict(noise=2.0, region=(140, 110, 500, 530))),
    "illumination": (_illumination_sequence, dict(noise=2.0, illum=True)),
    "noisy": (_noisy_sequence, dict(noise=5.1)),
    "nonuniform": (_nonuniform_sequence, dict(noise=2.0)),
}


def generate(preset: str, output: Path, n_frames: int, seed: int) -> Path:
    """Generate one demo sequence and write images + ground truth."""
    from PIL import Image

    builder, opts = PRESETS[preset]
    deformations, description = builder(n_frames)
    region = opts.get("region")
    pattern = DotPattern.generate(
        (SIZE, SIZE), n_dots=2400 if region is None else 1500, seed=seed, region=region
    )
    noise = float(opts.get("noise", 2.0))

    out_dir = output / preset
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for i, D in enumerate(deformations):
        kw = {}
        if opts.get("illum"):
            t = i / (n_frames - 1)
            kw = dict(gain=1.0 + 0.25 * t, offset=8.0 * t, illum_gradient=(0.02 * t, -0.015 * t))
        img = render(pattern, D, noise_sigma=noise, seed=seed * 1000 + i, **kw)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
            out_dir / f"image{i + 1:04d}.png"
        )
        images.append(img)

    # ground truth on a 32x32 grid
    g = np.linspace(20, SIZE - 20, 32)
    X, Y = np.meshgrid(g, g)
    xs, ys = X.ravel(), Y.ravel()
    rows = []
    for i, D in enumerate(deformations):
        gt = ground_truth_fields(D, xs, ys)
        for k in range(xs.size):
            rows.append({
                "frame": i, "X_px": xs[k], "Y_px": ys[k],
                "ux_px": gt["ux"][k], "uy_px": gt["uy"][k],
                "exx": gt["exx"][k], "eyy": gt["eyy"][k], "gamma_xy": gt["gamma_xy"][k],
                "Exx": gt["Exx"][k], "Eyy": gt["Eyy"][k], "Exy": gt["Exy"][k],
            })
    gt_path = out_dir / "ground_truth_grid.csv"
    with open(gt_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    meta = {
        "preset": preset,
        "description": description,
        "size_px": [SIZE, SIZE],
        "n_frames": n_frames,
        "noise_sigma_grey": noise,
        "seed": seed,
        "reference_image": "image0001.png",
        "has_background": region is not None,
        "coordinate_convention": (
            "x = column (right), y = row (down); ux = x_deformed - X, uy = y_deformed - Y; "
            "exx = du_x/dX, eyy = du_y/dY, gamma_xy = du_x/dY + du_y/dX (engineering, image frame)"
        ),
        "deformation_parameters": [getattr(d, "description", type(d).__name__) for d in deformations],
        "ground_truth": "ground_truth_grid.csv",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{preset}] {n_frames} frames -> {out_dir}")
    print(f"    {description}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="examples/demo_data", help="Output root directory.")
    p.add_argument("--preset", choices=tuple(PRESETS) + ("all",), default="all",
                   help="Which sequence(s) to generate (default: all).")
    p.add_argument("--frames", type=int, default=12, help="Number of frames per sequence.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (reproducible).")
    args = p.parse_args(argv)

    out = Path(args.output)
    presets = list(PRESETS) if args.preset == "all" else [args.preset]
    for preset in presets:
        generate(preset, out, args.frames, args.seed)
    print("\nRun the pipeline, e.g.:")
    print(f"  python -m dic {out / 'rotation'} /tmp/dic_demo_rotation --scale 0.05 --unit mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
