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


@pytest.mark.parametrize("dl0", [0.2, 0.05, 0.01])
def test_cal_surf_goda_point_matches_profile(dl0: float) -> None:
    """単一点計算が、沖から計算した断面の同じ水深の結果と一致すること。"""
    tant = 0.015
    h0l0 = 0.035

    df_profile = core.cal_surf_goda(tant, h0l0)
    df_point = core.cal_surf_goda_point(tant, h0l0, dl0)

    expected = df_profile[np.isclose(df_profile["dl0"], dl0)]
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        df_point,
        check_dtype=False,
        atol=1e-8,
        rtol=1e-6,
    )


def test_cal_surf_goda_point_includes_setup() -> None:
    """砕波帯内の単一点計算でもセットアップが考慮されること。"""
    df = core.cal_surf_goda_point(0.01, 0.02, 0.01)
    assert df["etal0"].iloc[0] > 0


def test_cal_surf_goda_order_independent() -> None:
    """dl0 の並び順によらず同じ結果を、入力と同じ順序で返すこと。"""
    tant = 0.01
    h0l0 = 0.02
    descending = [0.1, 0.05, 0.03, 0.02, 0.01]
    shuffled = [0.03, 0.01, 0.1, 0.02, 0.05]

    df_desc = core.cal_surf_goda(tant, h0l0, descending)
    df_shuf = core.cal_surf_goda(tant, h0l0, shuffled)

    np.testing.assert_allclose(df_shuf["dl0"].to_numpy(), shuffled)
    pd.testing.assert_frame_equal(
        df_shuf.set_index("dl0").sort_index(),
        df_desc.set_index("dl0").sort_index(),
    )
    # 最も岸側の点でセットアップ最大
    assert df_shuf["etal0"].idxmax() == df_shuf["dl0"].idxmin()


def test_cal_surf_goda_sparse_matches_default_profile() -> None:
    """少ない計算点を指定しても、デフォルトの細かい断面と同じ結果になること。"""
    tant = 0.01
    h0l0 = 0.02
    sparse = [0.1, 0.03, 0.01]

    df_full = core.cal_surf_goda(tant, h0l0)
    df_sparse = core.cal_surf_goda(tant, h0l0, sparse)

    expected = pd.concat(
        [df_full[np.isclose(df_full["dl0"], v)] for v in sparse], ignore_index=True
    )
    pd.testing.assert_frame_equal(df_sparse, expected)


def test_cal_surf_goda_duplicate_dl0() -> None:
    """dl0 に重複がある場合は同じ結果を返すこと。"""
    df = core.cal_surf_goda(0.01, 0.02, [0.1, 0.05, 0.05])
    assert len(df) == 3
    pd.testing.assert_series_equal(df.iloc[1], df.iloc[2], check_names=False)


def test_cal_surf_goda_rejects_empty_dl0() -> None:
    """dl0 が空の場合に ValueError を返すこと。"""
    with pytest.raises(ValueError):
        core.cal_surf_goda(0.01, 0.02, [])


def test_cal_surf_goda_fine_dl0_is_stable() -> None:
    """デフォルト点と近接する細かい dl0 を指定しても結果が発散しないこと。"""
    tant = 1 / 30
    h0l0 = 0.01
    dh0 = np.arange(4.0, 0.0, -0.05)

    df_fine = core.cal_surf_goda(tant, h0l0, list(dh0 * h0l0))
    df_full = core.cal_surf_goda(tant, h0l0)
    df_full = df_full[df_full["dh0"] <= 4.0].iloc[::-1]

    assert (df_fine["H1_3"] > 0).all()
    h13 = np.interp(df_full["dh0"], df_fine["dh0"].iloc[::-1], df_fine["H1_3"].iloc[::-1])
    np.testing.assert_allclose(h13, df_full["H1_3"], atol=0.05)


@pytest.mark.parametrize("h0l0", [0.002, 0.005, 0.01, 0.04])
def test_cal_surf_goda_deep_water_follows_rayleigh(h0l0: float) -> None:
    """沖側では H1/3 ≈ Ks・H0 となり、代表波高比がレイリー分布に従うこと。"""
    df = core.cal_surf_goda(1 / 30, h0l0, [1.0, 0.5])
    np.testing.assert_allclose(df["H1_3"], df["aks"], rtol=0.01)
    np.testing.assert_allclose(df["H1_250"] / df["H1_3"], 1.80, rtol=0.01)
    np.testing.assert_allclose(df["H1_1000"] / df["H1_3"], 1.98, rtol=0.01)


# 合田 (1975) 図-50〜57 から読み取った値 (海底勾配, H0'/L0, h/H0', 波高比)。
# 各曲線のピーク付近を 400 dpi のスキャン画像から読み取った。図-52 は図枠の下辺が方眼と
# ずれて描かれているため、方眼を基準に読み取った。
# sample/compare_goda1975.py で原図と計算結果を重ねて確認できる。
GODA1975_CHART_H13 = [
    (1 / 10, 0.002, 2.85, 2.53),
    (1 / 10, 0.005, 2.18, 1.90),
    (1 / 10, 0.01, 1.83, 1.54),
    (1 / 10, 0.02, 1.55, 1.26),
    (1 / 20, 0.002, 3.30, 2.23),
    (1 / 20, 0.005, 2.53, 1.68),
    (1 / 20, 0.01, 2.13, 1.37),
    (1 / 20, 0.02, 1.85, 1.13),
    (1 / 30, 0.002, 3.43, 2.16),
    (1 / 30, 0.005, 2.65, 1.63),
    (1 / 30, 0.01, 2.23, 1.33),
    (1 / 30, 0.02, 1.98, 1.10),
    (1 / 100, 0.002, 3.65, 2.04),
    (1 / 100, 0.005, 2.85, 1.55),
    (1 / 100, 0.01, 2.40, 1.27),
    (1 / 100, 0.02, 2.18, 1.06),
]
GODA1975_CHART_HMAX = [
    (1 / 10, 0.002, 2.55, 3.42),
    (1 / 10, 0.005, 2.63, 3.04),
    (1 / 10, 0.01, 2.23, 2.47),
    (1 / 10, 0.02, 1.93, 2.03),
    (1 / 20, 0.01, 2.60, 2.26),
    (1 / 30, 0.005, 3.23, 2.60),
    (1 / 30, 0.01, 2.78, 2.16),
    (1 / 30, 0.02, 2.63, 1.84),
    (1 / 100, 0.005, 3.48, 2.49),
    (1 / 100, 0.01, 3.05, 2.08),
    (1 / 100, 0.02, 2.83, 1.81),
]


@pytest.mark.parametrize("tant,h0l0,dh0,expected", GODA1975_CHART_H13)
def test_cal_surf_goda_matches_goda1975_chart_h13(
    tant: float, h0l0: float, dh0: float, expected: float
) -> None:
    """有義波高が合田 (1975) の算定図（図-50〜53）と一致すること。"""
    df = core.cal_surf_goda(tant, h0l0, [dh0 * h0l0])
    assert df["H1_3"].iloc[0] == pytest.approx(expected, abs=0.04)


@pytest.mark.parametrize("tant,h0l0,dh0,expected", GODA1975_CHART_HMAX)
def test_cal_surf_goda_matches_goda1975_chart_hmax(
    tant: float, h0l0: float, dh0: float, expected: float
) -> None:
    """最高波高 (Hmax = H1/250) が合田 (1975) の算定図（図-54〜57）と一致すること。"""
    df = core.cal_surf_goda(tant, h0l0, [dh0 * h0l0])
    assert df["H1_250"].iloc[0] == pytest.approx(expected, abs=0.04)


def test_cal_surf_goda_matches_expected_values() -> None:
    tant = 0.01
    h0l0 = 0.02
    dl0_values = [0.4, 0.3, 0.2]

    df = core.cal_surf_goda(tant, h0l0, dl0_values).round(6)

    expected = pd.DataFrame(
        {
            "H1_1000": [1.937709, 1.883937, 1.822652],
            "H1_250": [1.759616, 1.710785, 1.655134],
            "H1_100": [1.628846, 1.583645, 1.532129],
            "H1_50": [1.523684, 1.481401, 1.433211],
            "H1_10": [1.242032, 1.207565, 1.168283],
            "H1_5": [1.098329, 1.06785, 1.033113],
            "H1_3": [0.977155, 0.950039, 0.919134],
            "dh0": [20.0, 15.0, 10.0],
            "dl0": [0.4, 0.3, 0.2],
            "etal0": [0.000001, 0.000004, 0.000008],
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


def test_cal_surf_goda_representative_heights_ordered() -> None:
    """代表波高が超過確率の順に単調であること。

    H1/1000 >= H1/250 >= H1/100 >= H1/50 >= H1/10 >= H1/5 >= H1/3
    """
    df = core.cal_surf_goda(0.01, 0.02, [0.4, 0.3, 0.2, 0.1, 0.05])
    cols = ["H1_1000", "H1_250", "H1_100", "H1_50", "H1_10", "H1_5", "H1_3"]
    for hi, lo in zip(cols[:-1], cols[1:], strict=False):
        assert (df[hi] >= df[lo]).all(), f"{hi} < {lo} となる行があります"


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
    assert "H1_1000_dim" in df.columns
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
    # 浅い順に指定しても、岸側ほどセットアップが大きい
    assert df["etal0"].iloc[0] > df["etal0"].iloc[-1]
    assert df["etal0"].iloc[0] > 0


def test_cal_surf_goda_dim_numpy_scalar_depth() -> None:
    """numpy のスカラー水深も単一値として扱うこと。"""
    df_int = core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=10.0, d=np.int64(10), dim=True)
    df_float = core.cal_surf_goda_dim(tant=0.01, H0=5.0, T=10.0, d=10.0, dim=True)
    pd.testing.assert_frame_equal(df_int, df_float)


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


def test_bindx_uses_goda1975_breaker_index() -> None:
    """砕波限界が合田 (1975) 式 (3)・(8) (K=15, s=4/3, A1=0.18, A2=0.12) に従うこと。"""
    ddl0, h0l0, tant = 0.05, 0.02, 1 / 30
    xxx = (1.0 - np.exp(-1.5 * np.pi * ddl0 * (1.0 + 15.0 * tant ** (4.0 / 3.0)))) / h0l0
    x1, x2 = core.bindx(ddl0, h0l0, tant)
    assert x1 == pytest.approx(0.18 * xxx)
    assert x2 == pytest.approx(0.12 * xxx)


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
