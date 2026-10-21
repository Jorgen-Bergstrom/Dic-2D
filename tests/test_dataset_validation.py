"""Tests for the provided-dataset ground-truth loader."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from dic.dataset_validation import load_ground_truth_grid
from dic.kinematics import left_stretch


def _write_grid(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_ground_truth_grid_uses_nominal_strain_from_J(tmp_path):
    path = tmp_path / "ground_truth_grid.csv"
    _write_grid(
        path,
        [
            {
                "image_index": 0,
                "X_px": 0.0,
                "Y_px": 0.0,
                "ux_px": 0.0,
                "uy_px": 0.0,
                "eps_xx_img": 0.0,
                "eps_yy_img": 0.0,
                "gamma_xy_img": 0.5,
                "J11": 1.0,
                "J12": 0.5,
                "J21": 0.0,
                "J22": 1.0,
                "in_ref": 1,
                "in_def": 1,
            }
        ],
    )
    gt = load_ground_truth_grid(path)
    V11, V12, V22 = left_stretch(1.0, 0.5, 0.0, 1.0)
    assert gt[0]["Eeng_xx"][0] == pytest.approx(V11 - 1.0)
    assert gt[0]["Eeng_yy"][0] == pytest.approx(V22 - 1.0)
    assert gt[0]["Eeng_gamma_xy"][0] == pytest.approx(2.0 * V12)
    assert gt[0]["Eeng_gamma_xy"][0] != pytest.approx(0.5)


def test_ground_truth_grid_falls_back_to_engineering_columns(tmp_path):
    path = tmp_path / "ground_truth_grid.csv"
    _write_grid(
        path,
        [
            {
                "image_index": 0,
                "X_px": 0.0,
                "Y_px": 0.0,
                "ux_px": 0.0,
                "uy_px": 0.0,
                "eps_xx_img": 0.01,
                "eps_yy_img": -0.02,
                "gamma_xy_img": 0.03,
                "in_ref": 1,
                "in_def": 1,
            }
        ],
    )
    gt = load_ground_truth_grid(path)
    assert gt[0]["Eeng_xx"][0] == pytest.approx(0.01)
    assert gt[0]["Eeng_yy"][0] == pytest.approx(-0.02)
    assert gt[0]["Eeng_gamma_xy"][0] == pytest.approx(0.03)
    assert np.isnan(gt[0]["J11"][0])
