from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

import goda_wb.core as core
from goda_wb.constant import g, pi2

# ---------------------------------------------------------------------------
# wave()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", [0.5, 1.5, 8.0])
def test_wave_satisfies_dispersion_relation(d: float) -> None:
    """wave(d) が分散関係 x * tanh(x) = d を満たすかを確認する。"""
    result = core.wave(d)
    assert np.isclose(d, result * np.tanh(result), rtol=1e-5, atol=1e-6)


def test_wave_returns_identity_for_large_argument() -> None:
    """d が十分大きい場合は wave(d) が d をそのまま返すはず。"""
    assert core.wave(12.0) == pytest.approx(12.0)


@pytest.mark.parametrize("d", [0.01, 0.99, 1.0, 1.01, 9.99, 10.0, 10.01])
def test_wave_boundary_values(d: float) -> None:
    """分岐点付近での wave() の正確性。"""
    result = core.wave(d)
    if d > 10.0:
        assert result == pytest.approx(d)
    else:
        assert np.isclose(d, result * np.tanh(result), rtol=1e-4, atol=1e-6)


def test_wave_very_small_d() -> None:
    """d が非常に小さい場合の動作確認。"""
    result = core.wave(0.001)
    assert np.isclose(0.001, result * np.tanh(result), rtol=1e-3, atol=1e-6)


def test_wave_rejects_negative() -> None:
    """負の値に対して ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.wave(-1.0)


# ---------------------------------------------------------------------------
# aksi()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dl0", [0.01, 0.05, 0.1, 0.3, 0.5, 1.0])
def test_aksi_positive_result(dl0: float) -> None:
    """aksi() は正の値を返すこと。"""
    result = core.aksi(dl0)
    assert result > 0


def test_aksi_deep_water_approaches_one() -> None:
    """深水域では浅水係数が 1 に近づくこと。"""
    result = core.aksi(1.0)
    assert result == pytest.approx(1.0, abs=0.05)


def test_aksi_rejects_non_positive() -> None:
    """非正の値に対して ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.aksi(0.0)
    with pytest.raises(ValueError):
        core.aksi(-0.5)


# ---------------------------------------------------------------------------
# cal_wave_length()
# ---------------------------------------------------------------------------

def test_cal_wave_length_deep_water() -> None:
    """深水条件での波長が L0 = g*T^2/(2*pi) に近いこと。"""
    T = 10.0
    d = 100.0
    L0 = g * T**2 / pi2
    L = core.cal_wave_length(d, T)
    assert pytest.approx(L0, rel=0.01) == L


def test_cal_wave_length_positive() -> None:
    """波長は常に正であること。"""
    L = core.cal_wave_length(5.0, 8.0)
    assert L > 0


def test_cal_wave_length_shorter_in_shallow_water() -> None:
    """浅水域の波長は深水域より短いこと。"""
    T = 10.0
    L_deep = core.cal_wave_length(100.0, T)
    L_shallow = core.cal_wave_length(2.0, T)
    assert L_shallow < L_deep


@pytest.mark.parametrize("d,T", [(0, 10.0), (-5, 10.0), (10.0, 0), (10.0, -3)])
def test_cal_wave_length_rejects_invalid(d: float, T: float) -> None:
    """不正な入力に対して ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.cal_wave_length(d, T)


# ---------------------------------------------------------------------------
# shoal()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dl0", [0.5, 0.3, 0.1, 0.05, 0.01])
def test_shoal_positive_result(dl0: float) -> None:
    """shoal() は正の値を返すこと。"""
    result = core.shoal(dl0, 0.02)
    assert result > 0


def test_shoal_deep_water_matches_aksi() -> None:
    """深水域では shoal() と aksi() が一致すること。"""
    dl0 = 0.5
    h0l0 = 0.02
    aks = core.shoal(dl0, h0l0)
    ksi = core.aksi(dl0)
    assert aks == pytest.approx(ksi, rel=1e-6)


def test_shoal_rejects_non_positive() -> None:
    """非正の値に対して ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.shoal(0.0, 0.02)
    with pytest.raises(ValueError):
        core.shoal(0.1, 0.0)
    with pytest.raises(ValueError):
        core.shoal(0.1, -0.01)


def test_shoal_different_steepness() -> None:
    """異なる波形勾配で結果が異なること。"""
    dl0 = 0.05
    aks1 = core.shoal(dl0, 0.01)
    aks2 = core.shoal(dl0, 0.05)
    assert aks1 != pytest.approx(aks2, abs=1e-6)


# ---------------------------------------------------------------------------
# cal_surf_goda()
# ---------------------------------------------------------------------------

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


def test_cal_surf_goda_default_dl0() -> None:
    """dl0=None の場合にデフォルトリストが使用されること。"""
    from goda_wb.constant import goda_dl0_list

    df = core.cal_surf_goda(0.01, 0.02)
    assert len(df) == len(goda_dl0_list)


def test_cal_surf_goda_rejects_invalid() -> None:
    """不正な入力に対して ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.cal_surf_goda(-0.01, 0.02, [0.1])
    with pytest.raises(ValueError):
        core.cal_surf_goda(0.01, 0.0, [0.1])
    with pytest.raises(ValueError):
        core.cal_surf_goda(0.01, -0.02, [0.1])


def test_cal_surf_goda_h13_less_than_h1_1000() -> None:
    """H1/3 は常に H1/1000 以下であること。"""
    df = core.cal_surf_goda(0.01, 0.02, [0.3, 0.2, 0.1])
    assert (df["H1_3"] <= df["H1_1000"]).all()


# ---------------------------------------------------------------------------
# cal_surf_goda_dim()
# ---------------------------------------------------------------------------

def test_cal_surf_goda_dim_basic() -> None:
    """基本的な有次元計算が正しく動作すること。"""
    df = core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=10.0)
    assert len(df) > 0
    assert "H1_3" in df.columns


def test_cal_surf_goda_dim_with_dim_flag() -> None:
    """dim=True の場合に有次元カラムが追加されること。"""
    df = core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=10.0, dim=True)
    assert "d" in df.columns
    assert "H1_3_dim" in df.columns
    assert "H1_250_dim" in df.columns
    assert "H1_100_dim" in df.columns
    assert "H1_50_dim" in df.columns
    assert "H1_10_dim" in df.columns
    assert "H1_5_dim" in df.columns


def test_cal_surf_goda_dim_single_depth() -> None:
    """単一水深を指定した場合の動作確認。"""
    df = core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=10.0, d=10.0, dim=True)
    assert len(df) == 1
    assert df["d"].iloc[0] == pytest.approx(10.0, rel=0.01)


def test_cal_surf_goda_dim_depth_list() -> None:
    """水深リストを指定した場合の動作確認。"""
    depths = [5.0, 10.0, 15.0]
    df = core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=10.0, d=depths, dim=True)
    assert len(df) == len(depths)
    np.testing.assert_allclose(df["d"].to_numpy(), depths, rtol=0.01)


def test_cal_surf_goda_dim_h13_dim_scaling() -> None:
    """H1/3_dim = H1/3 * H0 であること。"""
    H0 = 5.0
    df = core.cal_surf_goda_dim(tant=0.01, H0=H0, T=10.0, dim=True, d=10.0)
    np.testing.assert_allclose(
        df["H1_3_dim"].to_numpy(),
        df["H1_3"].to_numpy() * H0,
        rtol=1e-10,
    )


def test_cal_surf_goda_dim_rejects_invalid() -> None:
    """不正な入力に対して ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.cal_surf_goda_dim(tant=0.01, H0=0, T=10.0)
    with pytest.raises(ValueError):
        core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=0)
    with pytest.raises(ValueError):
        core.cal_surf_goda_dim(tant=0.01, H0=-1, T=10.0)
    with pytest.raises(ValueError):
        core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=-1)


# ---------------------------------------------------------------------------
# sbeat()
# ---------------------------------------------------------------------------

def test_sbeat_rejects_invalid_j() -> None:
    """j が範囲外の場合に ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.sbeat(0.02, 0.1, -1)
    with pytest.raises(ValueError):
        core.sbeat(0.02, 0.1, 8)


def test_sbeat_returns_valid_probability() -> None:
    """sbeat() の確率が正であること。"""
    for j in range(8):
        _, pxi = core.sbeat(0.02, 0.1, j)
        assert pxi > 0
