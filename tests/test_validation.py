"""Integration test for the synthetic validation suite (requirement 5.5b).

This is the slow test of the suite: it runs every synthetic case and checks that the report is
written and that every case meets its pre-stated criterion.  It is marked ``slow`` but still runs
by default, because a failing validation is a failing build.
"""

from __future__ import annotations

import pytest

from dic.validation import build_cases, run_validation


@pytest.mark.slow
def test_validation_suite_passes_in_quick_mode(tmp_path):
    path, cases = run_validation(tmp_path, quick=True)
    assert path.is_file()
    text = path.read_text()
    assert "Part A" in text and "Part B" in text
    assert len(cases) == 11
    failed = [c.key for c in cases if not c.passed]
    assert not failed, f"validation cases failed: {failed}\n{text}"
    # every case must report a finite u RMS and a coverage
    import numpy as np

    for c in cases:
        assert np.isfinite(c.metrics["u_rms"])
        assert c.metrics["coverage"] > 0.3


def test_build_cases_have_criteria_and_ground_truth():
    cases = build_cases(quick=True)
    keys = {c.key for c in cases}
    assert {
        "zero_motion",
        "translation_subpixel",
        "translation_multipixel",
        "rotation_30_incremental",
        "rotation_30_single_step",
        "stretch_20",
        "shear_affine",
        "nonuniform",
        "illumination",
        "masked_background",
        "noise_2pct",
    } <= keys
    for c in cases:
        assert c.criterion
        assert len(c.images) == len(c.deformations) >= 2
