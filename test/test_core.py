from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

import goda_wb.core as core


@pytest.mark.parametrize("d", [0.5, 1.5, 8.0])
def test_wave_satisfies_dispersion_relation(d: float) -> None:
    """wave(d) が分散関係 x * tanh(x) = d を満たすかを確認する。"""
    result = core.wave(d)
    assert np.isclose(d, result * np.tanh(result), rtol=1e-5, atol=1e-6)


def test_wave_returns_identity_for_large_argument() -> None:
    """d が十分大きい場合は wave(d) が d をそのまま返すはず。"""
    assert core.wave(12.0) == pytest.approx(12.0)


def test_cal_surf_goda_dataframe_shapes_and_values() -> None:
    tant = 0.01
    h0l0 = 0.02
    dl0_values = [0.4, 0.3, 0.2]

    df = core.cal_surf_goda(tant, h0l0, dl0_values)

    expected_columns = [
        "H1_1000",
        "H1_250",
        "H1_100",
        "H1_50",
        "H1_10",
        "H1_5",
        "H1_3",
        "dh0",
        "dl0",
        "etal0",
        "aks",
    ]

    assert list(df.columns) == expected_columns
    assert len(df) == len(dl0_values)
    np.testing.assert_allclose(df["dl0"].to_numpy(), dl0_values, rtol=0, atol=1e-12)

    expected_aks = [core.shoal(v, h0l0) for v in dl0_values]
    np.testing.assert_allclose(df["aks"].to_numpy(), expected_aks, rtol=1e-6, atol=1e-8)


def test_cal_surf_goda_point_matches_bulk_version() -> None:
    tant = 0.015
    h0l0 = 0.035
    dl0 = 0.21

    df_bulk = core.cal_surf_goda(tant, h0l0, [dl0])
    df_point = core.cal_surf_goda_point(tant, h0l0, dl0)

    pd.testing.assert_frame_equal(
        df_bulk.reset_index(drop=True),
        df_point.reset_index(drop=True),
        check_dtype=False,
        atol=1e-8,
        rtol=1e-6,
    )


def test_cal_surf_goda_matches_expected_values() -> None:
    tant = 0.01
    h0l0 = 0.02
    dl0_values = [0.4, 0.3, 0.2]

    df = core.cal_surf_goda(tant, h0l0, dl0_values).round(6)

    expected = pd.DataFrame(
        {
            "H1_1000": [1.959090, 1.901165, 1.838882],
            "H1_250": [1.785743, 1.732904, 1.667379],
            "H1_100": [1.290761, 1.255148, 1.209394],
            "H1_50": [1.536657, 1.498192, 1.445154],
            "H1_10": [1.255194, 1.222334, 1.175370],
            "H1_5": [1.110969, 1.077974, 1.040564],
            "H1_3": [0.987443, 0.957502, 0.923597],
            "dh0": [20.0, 15.0, 10.0],
            "dl0": [0.4, 0.3, 0.2],
            "etal0": [0.0, 0.000002, 0.000007],
            "aks": [0.976029, 0.948943, 0.918074],
        }
    )

    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        atol=5e-6,
        rtol=5e-6,
    )


