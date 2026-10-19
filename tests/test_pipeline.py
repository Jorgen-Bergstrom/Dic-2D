"""End-to-end tests of the pipeline, outputs, CLI and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dic.cli import build_parser, main
from dic.config import RunConfig
from dic.outputs import write_all
from dic.pipeline import run_pipeline
from dic.report import write_summary
from dic.synth import DotPattern, render, stretch


def _make_sequence(tmp_path: Path, n=4, size=180, noise=2.0, seed=0) -> Path:
    pat = DotPattern.generate((size, size), n_dots=300, seed=seed)
    d = tmp_path / "seq"
    d.mkdir(exist_ok=True)
    for i in range(n):
        D = stretch(0.06 * i, -0.02 * i, np.array([size / 2, size / 2]))
        img = render(pat, D, noise_sigma=noise, seed=100 + i)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / f"image{i + 1:04d}.png")
    return d


def _fast_cfg(**kw) -> RunConfig:
    cfg = RunConfig()
    cfg.correlation.subset_size = 21
    cfg.correlation.step = 10
    cfg.output.figures = False
    cfg.output.extras = False
    cfg.uncertainty.noise_floor = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_pipeline_runs_and_produces_expected_fields(tmp_path):
    d = _make_sequence(tmp_path)
    res = run_pipeline(d, _fast_cfg())
    assert res.n_frames == 4
    assert res.xs.size > 50
    assert res.mask.shape == (180, 180)
    f0 = res.frames[0]
    assert np.allclose(f0.ux, 0.0, atol=1e-6)
    # last frame: the mean exx should be close to 0.18
    last = res.frames[-1]
    assert np.nanmean(last.strain["exx"]) == pytest.approx(0.18, abs=5e-3)
    assert np.nanmean(last.strain["eyy"]) == pytest.approx(-0.06, abs=5e-3)


def test_pipeline_is_reproducible(tmp_path):
    d = _make_sequence(tmp_path)
    a = run_pipeline(d, _fast_cfg())
    b = run_pipeline(d, _fast_cfg())
    assert np.allclose(a.frames[-1].ux, b.frames[-1].ux, equal_nan=True)
    assert np.allclose(a.frames[-1].strain["exx"], b.frames[-1].strain["exx"], equal_nan=True)


def test_pipeline_requires_enough_points(tmp_path):
    d = _make_sequence(tmp_path, size=60)
    cfg = _fast_cfg()
    cfg.correlation.subset_size = 51
    cfg.correlation.step = 40
    with pytest.raises(RuntimeError):
        run_pipeline(d, cfg)


def test_outputs_are_written(tmp_path):
    d = _make_sequence(tmp_path)
    res = run_pipeline(d, _fast_cfg())
    out = tmp_path / "out"
    write_all(res, out)
    write_summary(res, out)
    assert (out / "params.json").is_file()
    assert (out / "summary_report.md").is_file()
    assert (out / "displacement_timeseries.csv").is_file()
    assert (out / "strain_timeseries.csv").is_file()
    assert (out / "mask" / "mask_used.png").is_file()
    assert len(list((out / "fields").glob("*.npz"))) == res.n_frames
    params = json.loads((out / "params.json").read_text())
    assert params["grid"]["n_points"] == res.xs.size
    assert params["input"]["n_images"] == res.n_frames
    assert all(len(f["sha256"]) == 64 for f in params["input"]["files"])
    # a per-frame archive must contain the documented arrays
    npz = np.load(out / "fields" / "frame_0004.npz")
    for key in ("ux", "uy", "valid", "zncc", "sigma_u", "strain_exx", "strain_Exx"):
        assert key in npz


def test_cli_help_lists_options_and_defaults(capsys):
    parser = build_parser()
    text = parser.format_help()
    for token in ("--subset", "--step", "--strain-window", "--mask", "--dt", "--validate"):
        assert token in text
    assert "default" in text


def test_cli_end_to_end(tmp_path):
    d = _make_sequence(tmp_path)
    out = tmp_path / "cli_out"
    code = main([
        str(d), str(out),
        "--subset", "21", "--step", "10",
        "--no-figures", "--no-extras", "--no-noise-floor", "--quiet",
    ])
    assert code == 0
    assert (out / "summary_report.md").is_file()


def test_cli_missing_directory_returns_input_error(tmp_path):
    code = main([str(tmp_path / "does_not_exist"), str(tmp_path / "o"), "--quiet",
                 "--no-figures", "--no-extras", "--no-noise-floor"])
    assert code == 3


def test_cli_coverage_collapse_returns_error(tmp_path):
    d = _make_sequence(tmp_path)
    out = tmp_path / "cli_cov"
    code = main([
        str(d), str(out),
        "--subset", "21", "--step", "10",
        "--min-coverage", "0.999",
        "--no-figures", "--no-extras", "--no-noise-floor", "--quiet",
    ])
    assert code == 4


def test_cli_user_mask_path(tmp_path):
    d = _make_sequence(tmp_path)
    m = np.zeros((180, 180), np.uint8)
    m[40:140, 40:140] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(m).save(mask_path)
    out = tmp_path / "cli_mask"
    code = main([
        str(d), str(out), "--mask", str(mask_path), "--mask-mode", "user",
        "--subset", "21", "--step", "10",
        "--no-figures", "--no-extras", "--no-noise-floor", "--quiet",
    ])
    assert code == 0
    res = run_pipeline(d, _fast_cfg(), user_mask_path=mask_path)
    assert res.xs.min() >= 40 + 10


def test_incremental_reference_mode_runs(tmp_path):
    d = _make_sequence(tmp_path)
    cfg = _fast_cfg()
    cfg.correlation.reference = "incremental"
    cfg.correlation.update_interval = 2
    res = run_pipeline(d, cfg)
    assert res.n_frames == 4
    last = res.frames[-1]
    assert last.coverage > 0.5
    # accumulated total displacement must still grow with the applied stretch
    assert np.nanmax(np.abs(last.ux)) > 0.1
    assert np.nanmean(last.strain["exx"]) == pytest.approx(0.18, abs=2e-2)


def test_demo_generator_produces_usable_dataset(tmp_path):
    import subprocess
    import sys as _sys

    script = Path(__file__).resolve().parents[1] / "examples" / "generate_demo.py"
    out = tmp_path / "demo"
    subprocess.run(
        [_sys.executable, str(script), "--output", str(out), "--preset", "rotation", "--frames", "3"],
        check=True, capture_output=True,
    )
    assert (out / "rotation" / "image0001.png").is_file()
    assert (out / "rotation" / "ground_truth_grid.csv").is_file()
    res = run_pipeline(out / "rotation", _fast_cfg())
    assert res.n_frames == 3
