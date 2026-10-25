from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

import goda_wb.core as core
import goda_wb.pegbis as pegbis
from goda_wb.constant import PEGBIS_TP_T13

# ---------------------------------------------------------------------------
# wave_levels() / spectral_periods()
# ---------------------------------------------------------------------------


def test_wave_levels_reproduce_rayleigh_statistics() -> None:
    """上位 1/3 の平均が H0'、上位 1/250 の平均が 1.80 H1/3 となること（論文 2 章 (5)）。"""
    levels = pegbis.wave_levels(1.0)
    rep = pegbis._representative(levels)
    assert len(levels) == 831
    assert np.all(np.diff(levels) < 0)
    assert rep["H1_3"] == pytest.approx(1.0, abs=0.01)
    assert rep["H1_250"] / rep["H1_3"] == pytest.approx(1.80, abs=0.02)


def test_wave_levels_h1_floor_for_small_m() -> None:
    """M を減らしても最大レベルは 1.80 H0' を下回らないこと（論文 6 章）。"""
    assert pegbis.wave_levels(1.0, 61)[0] == pytest.approx(1.80)
    assert pegbis.wave_levels(1.0)[0] == pytest.approx(0.706 * np.sqrt(np.log(1662.0)))


@pytest.mark.parametrize(("M", "expected"), [(831, 0.080), (300, 0.080), (101, 0.077), (31, 0.074)])
def test_default_cb(M: int, expected: float) -> None:
    """波高レベル数に応じた Cb（論文 2 章 (5)）。"""
    assert pegbis.default_cb(M) == pytest.approx(expected)


def test_peak_period_ratio_matches_paper() -> None:
    """Tp/T1/3 が論文付録の Tp = 9.10 s, T1/3 = 8.01 s と一致すること。"""
    assert pytest.approx(9.10 / 8.01, rel=0.002) == PEGBIS_TP_T13


def test_spectral_periods() -> None:
    """N = 1 では T のみ、N > 1 では短周期から長周期へ並ぶこと。"""
    assert pegbis.spectral_periods(10.0, 1) == pytest.approx([10.0])
    periods = pegbis.spectral_periods(10.0, 10)
    assert len(periods) == 10
    assert np.all(np.diff(periods) > 0)


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (pegbis.wave_levels, (0.0,)),
        (pegbis.wave_levels, (1.0, 0)),
        (pegbis.spectral_periods, (0.0, 1)),
        (pegbis.spectral_periods, (10.0, 0)),
    ],
)
def test_level_and_period_reject_invalid(func, args) -> None:
    """不正な入力で ValueError となること。"""
    with pytest.raises(ValueError):
        func(*args)


# ---------------------------------------------------------------------------
# gradational_breaker_index()
# ---------------------------------------------------------------------------

H1_LEVEL = pegbis.wave_levels(1.0)[0]


# 図-3 (h/L0 = 0.01, 水平ステップ) と図-2 (I = 1/20, H0/L0 = 0.04) から読み取った値
@pytest.mark.parametrize(
    ("h", "L0", "slope", "hm", "expected"),
    [
        (0.25, 25.0, 0.0, 1.0, 0.39),
        (0.5, 50.0, 0.0, 1.0, 0.335),
        (1.0, 100.0, 0.0, 1.0, 0.31),
        (2.0, 25.0, 0.05, 1.0, 0.29),
        (2.0, 25.0, 0.05, 1.92, 0.37),
        (0.5, 25.0, 0.05, 1.0, 0.39),
    ],
)
def test_breaker_index_matches_paper_figures(
    h: float, L0: float, slope: float, hm: float, expected: float
) -> None:
    """式 (7)/(11) の κm が論文の図-2, 3 と一致すること。"""
    kappa = pegbis.gradational_breaker_index(h, L0, slope, hm, H1_LEVEL)
    assert kappa == pytest.approx(expected, abs=0.02)


# 図-1 (I = 0) の最大波高レベルの Hb/h = 2κ1 を h/L0 = 0.01 で読み取った値。
# 図中の H0/L0 は H1/L0 として描かれている
@pytest.mark.parametrize(("h1l0", "expected"), [(0.01, 0.92), (0.02, 1.02), (0.04, 1.18)])
def test_breaker_index_matches_fig1(h1l0: float, expected: float) -> None:
    """最大レベルの砕波限界波高比が論文の図-1 と一致すること。"""
    kappa = pegbis.gradational_breaker_index(0.01, 1.0, 0.0, h1l0, h1l0, cb2=0.080)
    assert 2.0 * kappa == pytest.approx(expected, abs=0.03)


def test_breaker_index_custom_cb() -> None:
    """Cb, Cb' を指定すると勾配の正負に応じて使い分けること。"""
    base = pegbis.gradational_breaker_index(1.0, 50.0, 0.02, 1.0, 1.0)
    low = pegbis.gradational_breaker_index(1.0, 50.0, 0.02, 1.0, 1.0, cb=0.070)
    flat = pegbis.gradational_breaker_index(1.0, 50.0, 0.0, 1.0, 1.0, cb=0.070)
    flat2 = pegbis.gradational_breaker_index(1.0, 50.0, 0.0, 1.0, 1.0, cb2=0.0613)
    assert low < base
    assert flat == pytest.approx(pegbis.gradational_breaker_index(1.0, 50.0, 0.0, 1.0, 1.0))
    assert flat2 < flat


def test_breaker_index_capped_beyond_dl0_02() -> None:
    """h/L0 > 0.2 では h/L0 = 0.2 の値を用いること。"""
    k02 = pegbis.gradational_breaker_index(0.2, 1.0, 0.02, 0.02, 0.04)
    k05 = pegbis.gradational_breaker_index(0.5, 1.0, 0.02, 0.02, 0.04)
    assert k05 == pytest.approx(k02)


def test_breaker_index_decreases_with_level() -> None:
    """波高の小さいレベルほど κm が小さいこと。"""
    levels = pegbis.wave_levels(1.0)
    kappa = pegbis.gradational_breaker_index(2.0, 50.0, 0.02, levels, levels[0])
    assert np.all(np.diff(kappa) < 0)


# ---------------------------------------------------------------------------
# 減衰係数
# ---------------------------------------------------------------------------


def test_breaking_decay_only_above_limit() -> None:
    """a < κh では 0、a > κh で正となり、α_b は海底勾配とともに増えること。"""
    a = np.array([0.1, 0.3])
    fd = pegbis.breaking_decay_coef(a, 1.0, 0.2, 0.02)
    assert fd[0] == 0.0
    assert fd[1] > 0.0
    steep = pegbis.breaking_decay_coef(a, 1.0, 0.2, 0.1)
    assert steep[1] > fd[1]
    assert pegbis.breaking_decay_coef(a, 1.0, 0.2, -0.05)[1] == pytest.approx(
        pegbis.breaking_decay_coef(a, 1.0, 0.2, 0.0)[1]
    )


def test_bottom_friction_shallow_limit() -> None:
    """kh → 0 で f_Df = fw a / (3π h²)（No.3 付録 式 (A.9)）となること。"""
    fd = pegbis.bottom_friction_coef(0.5, 2.0, 1.0e-4, 0.15)
    assert fd == pytest.approx(0.15 * 0.5 / (3.0 * np.pi * 4.0), rel=1e-4)


def test_bottom_friction_vanishes_in_deep_water() -> None:
    """深海では底面摩擦による減衰がほぼ 0 となること。"""
    assert pegbis.bottom_friction_coef(0.5, 100.0, 50.0, 0.01) == pytest.approx(0.0, abs=1e-15)


def test_swart_fw() -> None:
    """Swart の式と上限値 0.30。"""
    assert pegbis.swart_fw(0.7, 1.0) == pytest.approx(0.30)
    expected = np.exp(-5.977 + 5.213 * 0.01**0.194)
    assert pegbis.swart_fw(0.01, 1.0) == pytest.approx(expected)
    with pytest.raises(ValueError):
        pegbis.swart_fw(0.0, 1.0)


# ---------------------------------------------------------------------------
# cal_surf_pegbis_slope(): 一様勾配斜面
# ---------------------------------------------------------------------------

DH0 = [3.0, 2.0, 1.5, 1.0, 0.5]

# 論文 付図-4~7（Ver.5 の PEGBIS、M = 831, 一方向波, 周波数成分 10）の記号を
# 300 dpi のページ画像からデジタイズした値。h/H0' = 0.5, 1.0, …, 3.5 に対する H/H0'
# （記号の判別ができなかった点は None）
PEGBIS_DH0 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
PEGBIS_FIGS = {
    1 / 10: {
        ("H1_250", 0.02): [0.90, 1.36, 1.68, 1.86, 1.84, 1.80, 1.77],
        ("H1_250", 0.04): [0.80, 1.19, 1.47, 1.63, 1.67, 1.68, 1.69],
        ("H1_3", 0.02): [0.64, None, 1.09, 1.06, 1.02, 1.00, 0.98],
        ("H1_3", 0.04): [0.57, 0.85, 0.95, 0.95, 0.94, None, None],
    },
    1 / 20: {
        ("H1_250", 0.02): [None, 1.08, 1.44, 1.72, 1.83, 1.79, 1.77],
        ("H1_250", 0.04): [None, 0.97, 1.27, 1.51, 1.64, 1.67, 1.69],
        ("H1_3", 0.02): [None, 0.81, 1.04, 1.05, 1.02, 1.00, 0.98],
        ("H1_3", 0.04): [None, 0.73, 0.92, 0.95, 0.94, None, None],
    },
    1 / 30: {
        ("H1_250", 0.02): [None, 0.99, 1.34, 1.66, 1.82, 1.79, 1.77],
        ("H1_250", 0.04): [None, 0.90, 1.20, 1.47, 1.63, 1.66, 1.69],
        ("H1_3", 0.02): [None, 0.75, 1.00, 1.05, 1.03, 1.00, 0.98],
        ("H1_3", 0.04): [None, 0.69, 0.90, 0.94, 0.94, 0.93, None],
    },
    1 / 100: {
        ("H1_250", 0.02): [None, 0.89, 1.22, 1.53, 1.80, 1.78, 1.76],
        ("H1_250", 0.04): [None, None, 1.10, 1.37, 1.60, 1.63, 1.68],
        ("H1_3", 0.02): [None, 0.68, 0.95, 1.04, 1.02, 1.00, 0.98],
        ("H1_3", 0.04): [None, 0.63, 0.86, None, 0.94, 0.93, None],
    },
}


@pytest.mark.parametrize("tant", list(PEGBIS_FIGS))
@pytest.mark.parametrize("h0l0", [0.02, 0.04])
def test_slope_matches_paper_appendix_figures(tant: float, h0l0: float) -> None:
    """一様勾配斜面の Hmax, H1/3 が論文 付図-4~7 の計算値と 0.045 以内で一致すること。"""
    res = pegbis.cal_surf_pegbis_slope(tant, h0l0, [v * h0l0 for v in PEGBIS_DH0], n_freq=10)
    for (col, s), ref in PEGBIS_FIGS[tant].items():
        if s != h0l0:
            continue
        for value, expected in zip(res[col], ref, strict=True):
            if expected is not None:
                assert value == pytest.approx(expected, abs=0.045)


@pytest.mark.parametrize("tant", [1 / 30, 1 / 100])
@pytest.mark.parametrize("h0l0", [0.02, 0.04])
def test_slope_matches_goda1975_model(tant: float, h0l0: float) -> None:
    """勾配 1/30, 1/100 では合田 (1975) のモデルとよく一致すること（論文 図-8, 9）。"""
    dl0 = [v * h0l0 for v in DH0]
    res = pegbis.cal_surf_pegbis_slope(tant, h0l0, dl0)
    ref = core.cal_surf_goda(tant, h0l0, dl0)
    assert np.allclose(res["H1_3"], ref["H1_3"], atol=0.06)
    assert np.allclose(res["H1_250"], ref["H1_250"], atol=0.06)


def test_steep_slope_lower_than_goda1975_model() -> None:
    """勾配 1/10 では h/H0' = 1.5~2.5 で合田 (1975) のモデルより小さめとなること（図-6）。"""
    dl0 = [v * 0.02 for v in [2.0, 1.5]]
    res = pegbis.cal_surf_pegbis_slope(0.1, 0.02, dl0)
    ref = core.cal_surf_goda(0.1, 0.02, dl0)
    assert np.all(res["H1_3"] < ref["H1_3"])
    assert np.all(res["H1_250"] < ref["H1_250"])


@pytest.mark.parametrize(("n_freq", "expected"), [(1, 0.913), (10, 0.937)])
def test_minimum_shoaling_coefficient(n_freq: int, expected: float) -> None:
    """浅水係数の極小値: 単一周期 0.913、周波数スペクトル 0.937（論文 3 章 (2)）。"""
    dl0 = [0.2, 0.18, 0.16, 0.14, 0.12]
    res = pegbis.cal_surf_pegbis_slope(1 / 30, 0.001, dl0, n_freq=n_freq)
    assert res["aks"].min() == pytest.approx(expected, abs=0.003)


def test_representative_heights_ordered() -> None:
    """H1/250 >= H1/100 >= … >= H1/3 >= Hrms >= Hmean が常に成り立つこと。"""
    res = pegbis.cal_surf_pegbis_slope(1 / 20, 0.03)
    cols = ["H1_250", "H1_100", "H1_50", "H1_10", "H1_5", "H1_3", "Hrms", "Hmean"]
    vals = res[cols].to_numpy()
    assert np.all(np.diff(vals, axis=1) <= 1e-12)


def test_slope_dl0_order_and_duplicates() -> None:
    """dl0 の並び順は任意で、重複には同じ結果を返すこと。"""
    a = pegbis.cal_surf_pegbis_slope(1 / 30, 0.02, [0.01, 0.03, 0.01])
    b = pegbis.cal_surf_pegbis_slope(1 / 30, 0.02, [0.03, 0.01])
    assert list(a["dl0"]) == [0.01, 0.03, 0.01]
    assert a.loc[0, "H1_3"] == pytest.approx(b.loc[1, "H1_3"])
    assert a.loc[2, "H1_3"] == pytest.approx(a.loc[0, "H1_3"])


@pytest.mark.parametrize(
    ("tant", "h0l0", "dl0"),
    [(0.0, 0.02, None), (0.02, 0.0, None), (0.02, 0.02, []), (0.02, 0.02, [0.0])],
)
def test_slope_rejects_invalid(tant, h0l0, dl0) -> None:
    """不正な入力で ValueError となること。"""
    with pytest.raises(ValueError):
        pegbis.cal_surf_pegbis_slope(tant, h0l0, dl0)


# ---------------------------------------------------------------------------
# cal_surf_pegbis(): 任意断面
# ---------------------------------------------------------------------------

# 権の実験地形（論文 図-11）: 1/10 勾配 2 m、1/20 勾配 1 m、水深 0.11 m の水平床
KWEON_X = [-4.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
KWEON_H = [0.36, 0.36, 0.16, 0.11, 0.11, 0.11, 0.11, 0.11]


def test_offshore_boundary_is_linear_shoaling() -> None:
    """沖側境界の有義波高が H0' Ks となること。"""
    df = pegbis.cal_surf_pegbis([0.0, 100.0], [20.0, 15.0], 1.0, 8.0)
    assert df.loc[0, "H1_3"] == pytest.approx(df.loc[0, "Ks"], rel=0.01)


def test_horizontal_step_matches_paper_calculation() -> None:
    """権の水平ステップ実験（論文 図-11）の計算値を再現すること。

    図-11 は論文本文の図で、Ver.3 以前の α_b = 0.125 で計算されている。波高レベル数は論文の
    現地計算と同じ M = 301 とする。論文の計算値は、水平床上で Hmax ≈ 0.101, H1/10 ≈ 0.078,
    H1/3 ≈ 0.065 m、x = 0 で H1/3 ≈ 0.096, Hrms ≈ 0.071 m、Hmax の極大は x ≈ -1.1 m。
    沖側 (h = 0.36 m) の H1/3 は論文の 0.088 m に対し、浅水係数 Ks ≈ 1.03 の分だけ大きい。
    """
    x = np.linspace(-4.0, 4.0, 81)
    h = np.interp(x, KWEON_X, KWEON_H)
    df = pegbis.cal_surf_pegbis(x, h, 0.0879, 2.21, n_freq=10, alpha_b=0.125, M=301)
    flat = df[df["x"] >= 3.0]
    assert np.allclose(flat["H1_3"], flat["H1_3"].iloc[0], rtol=0.01)
    last = df.iloc[-1]
    assert last["H1_250"] == pytest.approx(0.101, rel=0.03)
    assert last["H1_10"] == pytest.approx(0.078, rel=0.03)
    assert last["H1_3"] == pytest.approx(0.065, rel=0.03)
    at0 = df[np.isclose(df["x"], 0.0)].iloc[0]
    assert at0["H1_3"] == pytest.approx(0.096, rel=0.03)
    assert at0["Hrms"] == pytest.approx(0.071, rel=0.03)
    assert -1.3 <= df.loc[df["H1_250"].idxmax(), "x"] <= -1.0
    assert df.loc[0, "H1_3"] == pytest.approx(0.088, rel=0.05)


def test_alpha_b_option() -> None:
    """α_b を小さくすると砕波帯内の波高が大きくなり、0 以下は ValueError となること。"""
    dl0 = [0.02, 0.01]
    new = pegbis.cal_surf_pegbis_slope(0.1, 0.02, dl0)
    old = pegbis.cal_surf_pegbis_slope(0.1, 0.02, dl0, alpha_b=0.125)
    assert np.all(old["H1_3"] > new["H1_3"])
    with pytest.raises(ValueError):
        pegbis.cal_surf_pegbis_slope(0.1, 0.02, dl0, alpha_b=0.0)


def test_friction_reduces_wave_height() -> None:
    """摩擦係数が大きいほどリーフ上の波高が小さくなること。"""
    x = [0.0, 50.0, 60.0, 300.0]
    h = [20.0, 3.0, 3.0, 3.0]
    smooth = pegbis.cal_surf_pegbis(x, h, 1.0, 8.0, fw=0.0)
    rough = pegbis.cal_surf_pegbis(x, h, 1.0, 8.0, fw=[0.0, 0.0, 0.15, 0.15])
    assert rough.iloc[-1]["H1_3"] < smooth.iloc[-1]["H1_3"]
    assert rough.iloc[1]["H1_3"] == pytest.approx(smooth.iloc[1]["H1_3"])


def test_result_independent_of_input_spacing() -> None:
    """入力点を追加しても同じ地形なら結果がほとんど変わらないこと。"""
    a = pegbis.cal_surf_pegbis([0.0, 400.0], [10.0, 2.0], 2.0, 8.0)
    b = pegbis.cal_surf_pegbis([0.0, 100.0, 250.0, 400.0], [10.0, 8.0, 5.0, 2.0], 2.0, 8.0)
    assert a.iloc[-1]["H1_3"] == pytest.approx(b.iloc[-1]["H1_3"], rel=0.01)


def test_lower_cb_increases_decay() -> None:
    """Cb を下げる（現地海岸向け 0.070）と砕波帯内の波高が小さくなること。"""
    x = [0.0, 300.0, 500.0]
    h = [10.0, 2.0, 2.0]
    lab = pegbis.cal_surf_pegbis(x, h, 3.0, 8.0)
    field = pegbis.cal_surf_pegbis(x, h, 3.0, 8.0, cb=0.070)
    assert field.iloc[-1]["H1_3"] < lab.iloc[-1]["H1_3"]
    assert field.iloc[0]["H1_3"] == pytest.approx(lab.iloc[0]["H1_3"])


@pytest.mark.parametrize(
    ("x", "h", "kwargs"),
    [
        ([0.0], [1.0], {}),
        ([0.0, 1.0], [1.0], {}),
        ([0.0, 0.0], [1.0, 1.0], {}),
        ([0.0, 1.0], [1.0, 0.0], {}),
        ([0.0, 1.0], [1.0, 1.0], {"fw": -0.1}),
        ([0.0, 1.0], [1.0, 1.0], {"cb": 0.0}),
    ],
)
def test_profile_rejects_invalid(x, h, kwargs) -> None:
    """不正な地形入力で ValueError となること。"""
    with pytest.raises(ValueError):
        pegbis.cal_surf_pegbis(x, h, 1.0, 8.0, **kwargs)
