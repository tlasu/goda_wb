"""段階的砕波変形モデル PEGBIS の岸沖方向 1 次元版。

合田良実 (2004)「不規則波浪変形計算における段階的砕波係数の導入について (Ver.5)」
ECOH/YG 技術論文 No.2 の PEGBIS (Parabolic Equation with Gradational Breaker Index for
Spectral waves) を、直角入射・一方向不規則波の岸沖方向 1 次元計算に絞って実装する。
直角入射では放物型方程式 (1) の屈折・回折項が消え、波高レベル m・周波数成分 n ごとの振幅 a は

    da/dx = -(1/2) a d(ln cg)/dx - f_D a,    f_D = f_Db + f_Df

に従う（微小振幅波の浅水変形と、砕波 (4)・底面摩擦 (14) による減衰）。
合田 (1975) のモデル (core) と異なり、任意の断面地形を扱えるが、ウェーブセットアップ・
サーフビート・非線形浅水変形は考慮しない。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from goda_wb.constant import (
    PEGBIS_BETA0,
    PEGBIS_BETA02,
    PEGBIS_BREAKING_DL0,
    PEGBIS_CB,
    PEGBIS_CB2,
    PEGBIS_CB_BY_LEVELS,
    PEGBIS_FW,
    PEGBIS_H1_MIN_RATIO,
    PEGBIS_KAPPA_SLOPE_COEF,
    PEGBIS_LEVELS,
    PEGBIS_MAX_DH_RATIO,
    PEGBIS_MAX_DX_H_RATIO,
    PEGBIS_MAX_DX_L_RATIO,
    PEGBIS_P,
    PEGBIS_P2,
    PEGBIS_SLOPE_DISTANCE_L0,
    PEGBIS_TP_T13,
    g,
    goda_dl0_list,
    pi2,
)
from goda_wb.core import wave

# 出力する代表波高と超過確率。Hmax は H1/250 で代表させる（論文 5 章）
_REPRESENTATIVE = {
    "H1_250": 1 / 250,
    "H1_100": 1 / 100,
    "H1_50": 1 / 50,
    "H1_10": 1 / 10,
    "H1_5": 1 / 5,
    "H1_3": 1 / 3,
}

# sinh のオーバーフローを避けるため、これより深い kh は深海波として扱う
_DEEP_KH = 20.0


def wave_levels(H0: float, M: int = PEGBIS_LEVELS) -> np.ndarray:
    """レイリー分布を等確率に M 分割した沖波の波高レベル（式 (5)）。

    論文のプログラムと同じく、M を減らしても最大の波高レベル H1 は 1.80 H0' を下回らない
    ようにする（6 章: 砕波帯外の沖側で Hmax = 1.80 H1/3 として計算を開始する）。

    Args:
        H0: 沖波の有義波高 H0' [m]
        M: 波高レベルの数

    Returns:
        大きい順に並んだ波高レベル H_m (m = 1, …, M) [m]
    """
    if H0 <= 0:
        raise ValueError(f"沖波波高は正の値でなければなりません: {H0}")
    if M < 1:
        raise ValueError(f"波高レベル数は 1 以上でなければなりません: {M}")
    m = np.arange(1, M + 1)
    levels = 0.706 * H0 * np.sqrt(np.log(2.0 * M / (2.0 * m - 1.0)))
    levels[0] = max(levels[0], PEGBIS_H1_MIN_RATIO * H0)
    return levels


def default_cb(M: int) -> float:
    """波高レベル数 M に応じた段階的砕波係数の定数 Cb（2 章 (5)）。

    M >= 300 では 0.080、M = 101, 61, 31 では 0.077, 0.076, 0.074 とし、間は線形補間する。

    Args:
        M: 波高レベルの数

    Returns:
        定数 Cb
    """
    ms, cbs = zip(*PEGBIS_CB_BY_LEVELS, strict=True)
    return float(np.interp(M, ms, cbs))


def spectral_periods(T: float, n_freq: int) -> np.ndarray:
    """エネルギーが等しくなるように分割した周波数成分の周期（式 (15)、γ = 1）。

    Args:
        T: 有義波周期 T1/3 [s]。ピーク周期は修正 Bretschneider-光易型スペクトルの
            関係 Tp = 1.136 T1/3 とする
        n_freq: 周波数成分の数 N。1 の場合は T のみを返す

    Returns:
        各成分の周期 [s]
    """
    if T <= 0:
        raise ValueError(f"周期は正の値でなければなりません: {T}")
    if n_freq < 1:
        raise ValueError(f"周波数成分の数は 1 以上でなければなりません: {n_freq}")
    if n_freq == 1:
        return np.array([T])
    fp = 1.0 / (PEGBIS_TP_T13 * T)
    n = np.arange(1, n_freq + 1)
    fn = 1.057 * fp * np.log(2.0 * n_freq / (2.0 * n_freq - 2.0 * n + 1.0)) ** (-0.25)
    return 1.0 / fn


def gradational_breaker_index(
    h: float,
    L0: float | np.ndarray,
    slope: float,
    Hm: float | np.ndarray,
    H1: float,
    cb: float = PEGBIS_CB,
    cb2: float = PEGBIS_CB2,
) -> float | np.ndarray:
    """段階的砕波係数 κm（式 (7)、I <= 0 では式 (11) の κm'）。

    κm は砕波限界振幅と水深の比で、振幅 a が κm h を超えると砕波減衰が働く。
    I <= 0 では定数を Cb', β0', p' に替え、I を含む項は I = 0 として評価する。
    β0 項の勾配係数は論文の印刷値 30 ではなく、論文の計算結果に合わせて較正した 60 を用いる
    （constant.PEGBIS_KAPPA_SLOPE_COEF）。
    h/L0 > 0.2 では h/L0 = 0.2 における値を用いる。

    Args:
        h: 水深 [m]
        L0: 深海波長 [m]
        slope: 海底勾配 I（岸向きに浅くなる向きを正）
        Hm: 沖波の波高レベル [m]
        H1: 沖波の最大の波高レベル [m]
        cb: I > 0 で用いる定数 Cb
        cb2: I <= 0 で用いる定数 Cb'

    Returns:
        段階的砕波係数 κm
    """
    if slope > 0:
        beta0, p, i = PEGBIS_BETA0, PEGBIS_P, slope
    else:
        cb, beta0, p, i = cb2, PEGBIS_BETA02, PEGBIS_P2, 0.0
    hh = np.minimum(h, PEGBIS_BREAKING_DL0 * L0)
    first = cb * L0 / hh * (1.0 - np.exp(-1.5 * np.pi * hh / L0 * (1.0 + 15.0 * i**2.5)))
    second = beta0 * Hm / hh * (Hm / L0) ** (-0.38) * np.exp(PEGBIS_KAPPA_SLOPE_COEF * i**2)
    return (first + second) * (Hm / H1) ** p


def breaking_decay_coef(
    a: float | np.ndarray,
    h: float,
    kappa: float | np.ndarray,
    slope: float,
    alpha_b: float | None = None,
) -> float | np.ndarray:
    """砕波による減衰係数 f_Db（式 (4)、α_b は付記の式 (付 1)）。

    Args:
        a: 振幅 [m]
        h: 水深 [m]
        kappa: 段階的砕波係数 κm
        slope: 海底勾配 I。α_b = 3/8 (0.3 + 2.5 I) の I は負のとき 0 とする
        alpha_b: 定数 α_b を直接与える場合の値。None の場合は上の勾配の関数（Ver.5）。
            論文本文の図（図-6~9, 11 など）は Ver.3 以前の一定値 0.125 で計算されている

    Returns:
        減衰係数 f_Db [1/m]
    """
    if alpha_b is None:
        alpha_b = 0.375 * (0.3 + 2.5 * max(slope, 0.0))
    ratio = a / (kappa * h)
    return np.where(ratio > 1.0, alpha_b / (2.0 * h) * np.sqrt(np.maximum(ratio**2 - 1.0, 0)), 0.0)


def bottom_friction_coef(
    a: float | np.ndarray, h: float, kh: float | np.ndarray, fw: float
) -> float | np.ndarray:
    """底面摩擦による減衰係数 f_Df（式 (14)）。

    Args:
        a: 振幅 [m]
        h: 水深 [m]
        kh: 波数と水深の積
        fw: 摩擦係数

    Returns:
        減衰係数 f_Df [1/m]
    """
    kh = np.minimum(kh, _DEEP_KH)
    shape = kh**2 / (np.sinh(kh) * (np.sinh(2.0 * kh) + 2.0 * kh))
    return 4.0 / (3.0 * np.pi) * fw * a / h**2 * shape


def swart_fw(ks: float, am: float) -> float:
    """Swart (1974) による乱流境界層の摩擦係数 fw。

    Args:
        ks: 底面の相当粗度 [m]
        am: 底面における水粒子の軌道振幅 [m]

    Returns:
        摩擦係数 fw。ks/am > 0.64 では上限値 0.30
    """
    if ks <= 0 or am <= 0:
        raise ValueError(f"ks と am は正の値でなければなりません: ks={ks}, am={am}")
    if ks / am > 0.64:
        return 0.30
    return float(np.exp(-5.977 + 5.213 * (ks / am) ** 0.194))


def _kh(h: float, L0: np.ndarray) -> np.ndarray:
    """各周波数成分の kh。"""
    return np.array([wave(pi2 * h / l0) for l0 in L0])


def _cg_ratio(kh: np.ndarray) -> np.ndarray:
    """群速度と深海波の波速の比 cg/C0。"""
    khc = np.minimum(kh, _DEEP_KH)
    return 0.5 * (1.0 + 2.0 * khc / np.sinh(2.0 * khc)) * np.tanh(khc)


def _fine_grid(x: np.ndarray, h: np.ndarray, L0: float) -> np.ndarray:
    """入力点を含み、水深変化と格子間隔の上限を満たす内部計算点。"""
    pts = [x[:1]]
    for k in range(len(x) - 1):
        dx = x[k + 1] - x[k]
        hmin = min(h[k], h[k + 1])
        lmin = L0 * float(_cg_ratio(np.array([wave(pi2 * hmin / L0)]))[0]) * 2.0
        n = max(
            1,
            int(np.ceil(abs(h[k + 1] - h[k]) / (PEGBIS_MAX_DH_RATIO * hmin))),
            int(np.ceil(dx / (PEGBIS_MAX_DX_H_RATIO * hmin))),
            int(np.ceil(dx / (PEGBIS_MAX_DX_L_RATIO * lmin))),
        )
        pts.append(np.linspace(x[k], x[k + 1], n + 1)[1:])
    return np.concatenate(pts)


def _local_slope(xf: np.ndarray, x: np.ndarray, h: np.ndarray, L0: float) -> np.ndarray:
    """沖側へ X = L0/10 離れた点との水深差から求めた海底勾配（2 章 (8)）。"""
    X = PEGBIS_SLOPE_DISTANCE_L0 * L0
    hf = np.interp(xf, x, h)
    xs = xf - X
    slope = np.empty_like(xf)
    inside = xs >= x[0]
    slope[inside] = (np.interp(xs[inside], x, h) - hf[inside]) / X
    near = ~inside & (xf > x[0])
    slope[near] = (h[0] - hf[near]) / (xf[near] - x[0])
    slope[xf <= x[0]] = (h[0] - h[1]) / (x[1] - x[0])
    return slope


def _representative(levels: np.ndarray) -> dict[str, float]:
    """等確率の波高レベルから代表波高を求める。"""
    hs = np.sort(levels)[::-1]
    M = len(hs)
    out = {}
    for name, q in _REPRESENTATIVE.items():
        k = q * M
        if k <= 1.0:
            out[name] = float(hs[0])
            continue
        nfull = int(np.floor(k))
        total = hs[:nfull].sum() + (k - nfull) * (hs[nfull] if nfull < M else 0.0)
        out[name] = float(total / k)
    out["Hrms"] = float(np.sqrt(np.mean(hs**2)))
    out["Hmean"] = float(np.mean(hs))
    return out


def cal_surf_pegbis(
    x: Sequence[float],
    h: Sequence[float],
    H0: float,
    T: float,
    fw: float | Sequence[float] = PEGBIS_FW,
    M: int = PEGBIS_LEVELS,
    n_freq: int = 1,
    cb: float | None = None,
    cb2: float | None = None,
    alpha_b: float | None = None,
) -> pd.DataFrame:
    """任意の断面地形における不規則波の砕波変形（PEGBIS の岸沖方向 1 次元版）。

    x[0] を沖側境界とし、そこでの波高は沖波から微小振幅波理論で浅水変形させた値とする
    （沖側境界では砕波していないことを前提とする）。入力点の間は水深を線形補間し、
    細かい内部格子で岸向きに積分する。周波数成分を分割した場合も、段階的砕波係数 κm の
    L0 には有義波周期の深海波長を用いる。

    Args:
        x: 岸向きに増加する距離 [m]
        h: 各点の水深 [m]。すべて正
        H0: 沖波の有義波高 H0' [m]
        T: 有義波周期 T1/3 [s]
        fw: 底面の摩擦係数。スカラーまたは x と同じ長さの配列（点間は線形補間）
        M: 波高レベルの数
        n_freq: 周波数成分の数。1 の場合は周期 T の単一成分、2 以上では式 (15) で分割
        cb: 段階的砕波係数の定数 Cb。None の場合は M に応じた値 (default_cb)。
            論文は現地海岸への適用には Cb = 0.070 を推奨している（9 章 (5)）
        cb2: 海底勾配 0 以下で用いる Cb'。None の場合は 0.080 × Cb / 0.080 の比例配分
        alpha_b: 砕波減衰の定数 α_b。None の場合は海底勾配の関数（付記の式 (付 1)）。
            論文本文の図を再現するときは Ver.3 以前の 0.125 を与える

    Returns:
        各入力点の結果を含む DataFrame。
        カラム: x, h, slope, Ks, H1_250, H1_100, H1_50, H1_10, H1_5, H1_3, Hrms, Hmean
        （Ks は各周波数成分の微小振幅波浅水係数のエネルギー平均）
    """
    x_arr = np.asarray(x, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    if x_arr.ndim != 1 or len(x_arr) < 2:
        raise ValueError("x は 2 点以上の 1 次元配列でなければなりません")
    if h_arr.shape != x_arr.shape:
        raise ValueError("x と h の長さが一致しません")
    if np.any(np.diff(x_arr) <= 0):
        raise ValueError("x は狭義単調増加でなければなりません")
    if np.any(h_arr <= 0):
        raise ValueError("水深は正の値でなければなりません")
    fw_arr = np.broadcast_to(np.asarray(fw, dtype=float), x_arr.shape)
    if np.any(fw_arr < 0):
        raise ValueError("摩擦係数は非負でなければなりません")

    if cb is None:
        cb = default_cb(M)
    if cb2 is None:
        cb2 = PEGBIS_CB2 * cb / PEGBIS_CB
    if alpha_b is not None and alpha_b <= 0:
        raise ValueError(f"α_b は正の値でなければなりません: {alpha_b}")
    if cb <= 0 or cb2 <= 0:
        raise ValueError(f"Cb, Cb' は正の値でなければなりません: cb={cb}, cb2={cb2}")
    levels = wave_levels(H0, M)
    periods = spectral_periods(T, n_freq)
    L0n = g * periods**2 / pi2
    L0 = g * T**2 / pi2

    xf = _fine_grid(x_arr, h_arr, L0)
    hf = np.interp(xf, x_arr, h_arr)
    fwf = np.interp(xf, x_arr, fw_arr)
    slope = _local_slope(xf, x_arr, h_arr, L0)

    kh_prev = _kh(hf[0], L0n)
    cg_prev = _cg_ratio(kh_prev)
    a = 0.5 * levels[:, None] * np.sqrt(0.5 / cg_prev)[None, :]

    def decay(amp: np.ndarray, j: int, kh: np.ndarray) -> np.ndarray:
        kappa = gradational_breaker_index(hf[j], L0, slope[j], levels[:, None], levels[0], cb, cb2)
        fd = breaking_decay_coef(amp, hf[j], kappa, slope[j], alpha_b)
        if fwf[j] > 0:
            fd = fd + bottom_friction_coef(amp, hf[j], kh[None, :], fwf[j])
        return fd

    out_index = {float(v): k for k, v in enumerate(x_arr)}
    rows: list[dict[str, float] | None] = [None] * len(x_arr)

    def record(j: int, amp: np.ndarray, cg: np.ndarray) -> None:
        k = out_index.get(float(xf[j]))
        if k is None:
            return
        level_h = np.sqrt(np.mean((2.0 * amp) ** 2, axis=1))
        row = {"x": float(xf[j]), "h": float(hf[j]), "slope": float(slope[j])}
        row["Ks"] = float(np.sqrt(np.mean(0.5 / cg)))
        row.update(_representative(level_h))
        rows[k] = row

    record(0, a, cg_prev)
    fd_prev = decay(a, 0, kh_prev)
    for j in range(1, len(xf)):
        dx = xf[j] - xf[j - 1]
        kh = _kh(hf[j], L0n)
        cg = _cg_ratio(kh)
        shoal = np.sqrt(cg_prev / cg)[None, :]
        # Heun 法: 前の点の減衰係数で予測し、両点の平均で修正する
        a_pred = a * shoal * np.exp(-fd_prev * dx)
        fd_pred = decay(a_pred, j, kh)
        a = a * shoal * np.exp(-0.5 * (fd_prev + fd_pred) * dx)
        fd_prev = decay(a, j, kh)
        cg_prev = cg
        record(j, a, cg)

    return pd.DataFrame(rows)


def cal_surf_pegbis_slope(
    tant: float,
    h0l0: float,
    dl0: Sequence[float] | None = None,
    fw: float = PEGBIS_FW,
    M: int = PEGBIS_LEVELS,
    n_freq: int = 1,
    cb: float | None = None,
    cb2: float | None = None,
    alpha_b: float | None = None,
) -> pd.DataFrame:
    """一様勾配斜面における PEGBIS の無次元計算（合田 (1975) のモデルとの比較用）。

    沖側境界を h/L0 = max(0.5, max(dl0)) とし、L0 = 1 の無次元地形で計算する。

    Args:
        tant: 海底勾配 (tan θ)。正の値
        h0l0: 沖波波形勾配 (H0'/L0)
        dl0: 水深波長比 (d/L0) のリスト。None の場合は goda_dl0_list の 0.5 以下の点
            （1e-5 未満を除く）。並び順は任意で、重複する値には同じ結果を返す
        fw: 底面の摩擦係数
        M: 波高レベルの数
        n_freq: 周波数成分の数
        cb: 段階的砕波係数の定数 Cb（cal_surf_pegbis と同じ）
        cb2: 海底勾配 0 以下で用いる Cb'（cal_surf_pegbis と同じ）
        alpha_b: 砕波減衰の定数 α_b（cal_surf_pegbis と同じ）

    Returns:
        各水深における波高比 (H/H0') を含む DataFrame。
        カラム: H1_250, H1_100, H1_50, H1_10, H1_5, H1_3, Hrms, Hmean, dh0, dl0, aks
    """
    if tant <= 0:
        raise ValueError(f"海底勾配は正の値でなければなりません: {tant}")
    if h0l0 <= 0:
        raise ValueError(f"波形勾配は正の値でなければなりません: {h0l0}")
    if dl0 is None:
        dl0 = [v for v in goda_dl0_list if 1.0e-5 <= v <= 0.5]
    targets = [float(v) for v in dl0]
    if not targets:
        raise ValueError("dl0 が空です")
    if min(targets) <= 0:
        raise ValueError("dl0 は正の値でなければなりません")

    start = max(0.5, max(targets))
    grid = sorted(set(targets) | {start}, reverse=True)
    x = [(start - v) / tant for v in grid]
    T = float(np.sqrt(pi2 / g))  # L0 = 1
    df = cal_surf_pegbis(
        x, grid, h0l0, T, fw=fw, M=M, n_freq=n_freq, cb=cb, cb2=cb2, alpha_b=alpha_b
    )

    cols = list(_REPRESENTATIVE) + ["Hrms", "Hmean"]
    res = df[cols] / h0l0
    res["dh0"] = df["h"] / h0l0
    res["dl0"] = df["h"]
    res["aks"] = df["Ks"]
    row = {v: k for k, v in enumerate(grid)}
    return res.iloc[[row[v] for v in targets]].reset_index(drop=True)
