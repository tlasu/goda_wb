from __future__ import annotations

import numpy as np
import pandas as pd

from goda_wb.constant import (
    CONVERGENCE_TOL,
    DEEP_WATER_THRESHOLD,
    GAUSS_HERMITE_ABSCISSAE,
    GAUSS_HERMITE_WEIGHTS,
    MP,
    NWAVE,
    REPRESENTATIVE_WAVE_PROBABILITIES,
    WAVE_CONVERGENCE_TOL,
    g,
    goda_dl0_list,
    pi2,
)


def aksi(dl0: float) -> float:
    """微小振幅波理論に基づく浅水係数の算出。

    Args:
        dl0: 水深沖波波長比 (d/L0)

    Returns:
        浅水係数 Ks
    """
    if dl0 <= 0:
        raise ValueError(f"dl0 は正の値でなければなりません: {dl0}")

    dl = wave(pi2 * dl0) / pi2
    sh4 = np.sinh(2.0 * pi2 * dl)
    th2 = np.tanh(pi2 * dl)
    ksi = 1.0 / np.sqrt((1.0 + 2.0 * pi2 * dl / sh4) * th2)
    return ksi


def cal_wave_length(d: float, T: float) -> float:
    """波長の算出。

    Args:
        d: 水深 [m]
        T: 周期 [s]

    Returns:
        波長 [m]
    """
    if d <= 0:
        raise ValueError(f"水深は正の値でなければなりません: {d}")
    if T <= 0:
        raise ValueError(f"周期は正の値でなければなりません: {T}")

    dl0 = d / (g * T**2 / pi2)
    return pi2 * d / wave(pi2 * dl0)


def wave(d: float) -> float:
    """微小振幅波の分散関係を反復法で解く。

    x * tanh(x) = d を満たす x を求める。

    Args:
        d: 2*pi*d/L0

    Returns:
        2*pi*d/L
    """
    if d < 0:
        raise ValueError(f"d は非負の値でなければなりません: {d}")

    if d > DEEP_WATER_THRESHOLD:
        return d

    x = np.sqrt(d) if d < 1.0 else d
    e = 99.9
    while np.abs(e) > WAVE_CONVERGENCE_TOL:
        cothx = 1.0 / np.tanh(x)
        xx = x - (x - d * cothx) / (1.0 + d * (cothx**2 - 1.0))
        e = 1.0 - xx / x
        x = xx
    return xx


def etai(i: int, dl0: list[float], etl0: list[float]) -> list[float]:
    """波高水位の線形外挿による初期値推定。

    Args:
        i: 現在の計算点インデックス
        dl0: 水深波長比のリスト
        etl0: セットアップ量のリスト（in-place 更新）

    Returns:
        更新された etl0
    """
    if i <= 1:
        etl0[i] = 0.0
    else:
        etl0[i] = etl0[i - 1] + (etl0[i - 1] - etl0[i - 2]) * (
            dl0[i] - dl0[i - 1]
        ) / (dl0[i - 1] - dl0[i - 2])
    return etl0


def sbeat(h0l0: float, dl0: float, j: int) -> tuple[float, float]:
    """サーフビートによる水位変動の算出。

    ガウス-エルミート求積法の点と重みを用いて、
    サーフビートによる水位変動量と確率を返す。

    Args:
        h0l0: 沖波波形勾配 (H0/L0)
        dl0: 水深波長比 (d/L0)
        j: ガウス求積点のインデックス (0-7)

    Returns:
        (xil0, pxi): 水位変動量と確率の組
    """
    if not 0 <= j < len(GAUSS_HERMITE_ABSCISSAE):
        n = len(GAUSS_HERMITE_ABSCISSAE) - 1
        raise ValueError(f"j は 0-{n} の範囲でなければなりません: {j}")

    xil0 = (
        GAUSS_HERMITE_ABSCISSAE[j]
        * 0.01
        * h0l0
        / np.sqrt(h0l0 * (1.0 + dl0 / h0l0))
    )
    pxi = GAUSS_HERMITE_WEIGHTS[j]
    return xil0, pxi


def bindx(ddl0: float, h0l0: float, tant: float) -> tuple[float, float]:
    """砕波限界指標の算出。

    合田の砕波限界式に基づき、砕波限界波高の上限・下限を計算する。

    Args:
        ddl0: セットアップ補正後の水深波長比
        h0l0: 沖波波形勾配 (H0/L0)
        tant: 海底勾配

    Returns:
        (x1, x2): 砕波限界の上限と下限（H/H0 の値）
    """
    pi = np.pi
    xxx = (1.0 - np.exp(-1.5 * pi * ddl0 * (1.0 + 11.0 * tant ** (4.0 / 3.0)))) / h0l0
    x1 = xxx * 0.18
    x2 = xxx * 0.12
    return x1, x2


def setup(
    etl0: list[float],
    dl0: list[float],
    h0l0: float,
    p: list[float],
    xp: list[float],
    i: int,
    h2l0: list[float],
) -> tuple[list[float], list[float]]:
    """波のセットアップ量の計算。

    波高分布から平均二乗波高を算出し、エネルギーフラックスの保存から
    セットアップ量を求める。

    Args:
        etl0: セットアップ量のリスト（in-place 更新）
        dl0: 水深波長比のリスト
        h0l0: 沖波波形勾配 (H0/L0)
        p: 波高確率密度のリスト
        xp: 波高比の離散点リスト
        i: 現在の計算点インデックス
        h2l0: 平均二乗波高のリスト（in-place 更新）

    Returns:
        (etl0, h2l0): 更新されたセットアップ量と平均二乗波高
    """
    mp = len(p)

    xxb = 0.0
    for k in range(mp - 1):
        xxb = xxb + (xp[k + 1] - xp[k]) / 12.0 * (
            (3.0 * p[k] + p[k + 1]) * xp[k] ** 2
            + 2.0 * (p[k] + p[k + 1]) * xp[k] * xp[k + 1]
            + (p[k] + 3.0 * p[k + 1]) * xp[k + 1] ** 2
        )
    h2l0[i] = xxb * h0l0 * h0l0
    if i == 0:
        etl0[i] = 0.0
        return etl0, h2l0

    dl1 = wave(pi2 * dl0[i - 1]) / pi2
    dl2 = wave(pi2 * dl0[i]) / pi2
    ake1 = 0.125 * (0.5 + 2.0 * pi2 * dl1 / np.sinh(2.0 * pi2 * dl1))
    ake2 = 0.125 * (0.5 + 2.0 * pi2 * dl2 / np.sinh(2.0 * pi2 * dl2))
    c1 = ake1 / (dl0[i - 1] + etl0[i - 1])
    c2 = ake2 / (dl0[i] + etl0[i])
    c = (c1 + c2) * 0.5

    etl0[i] = etl0[i - 1] - c * (h2l0[i] - h2l0[i - 1])

    return etl0, h2l0


def out(
    h0l0: float,
    dl0: list[float],
    etl0: list[float],
    p: list[float],
    xp: list[float],
    iflag: list[int],
    i: int,
) -> tuple[float, list[float]]:
    """代表波高の算出。

    波高確率密度分布から各超過確率に対応する代表波高比を計算する。

    Args:
        h0l0: 沖波波形勾配 (H0/L0)
        dl0: 水深波長比のリスト
        etl0: セットアップ量のリスト
        p: 波高確率密度のリスト
        xp: 波高比の離散点リスト
        iflag: フラグリスト
        i: 現在の計算点インデックス

    Returns:
        (dh0, hnh0): 水深波高比と各代表波高比のリスト
    """
    mp = MP
    hnh0 = [0.0] * NWAVE

    epn = REPRESENTATIVE_WAVE_PROBABILITIES

    dh0 = dl0[i] / h0l0

    ep = np.zeros(mp)
    hep = np.zeros(mp)

    for k in range(mp - 2, -1, -1):
        ep[k] = ep[k + 1] + 0.5 * (p[k + 1] + p[k]) * (xp[k + 1] - xp[k])
        hep[k] = 0.0
        for j in range(k, mp - 1, 1):
            hep[k] = hep[k] + (xp[j + 1] - xp[j]) / 6.0 * (
                (2.0 * p[j] + p[j + 1]) * xp[j] + (p[j] + 2.0 * p[j + 1]) * xp[j + 1]
            )
        if ep[k] != 0:
            hep[k] = hep[k] / ep[k]
        else:
            hep[k] = 0.0

    for n in range(NWAVE):
        found = False
        for k in range(mp - 2, -1, -1):
            if (ep[k + 1] < epn[n]) and (ep[k] >= epn[n]):
                found = True
                k1 = k + 1
                k2 = k
                ratk = (epn[n] - ep[k + 1]) / (ep[k] - ep[k + 1])
                break
        if not found:
            hnh0[n] = 0.0
        else:
            hnh0[n] = (1.0 - ratk) * hep[k1] + ratk * hep[k2]

    return dh0, hnh0


def prob(
    x1: float,
    x2: float,
    aks: float,
    j: int,
    pxi: float,
    xp: list[float],
    p: list[float],
) -> tuple[list[float], list[float]]:
    """砕波を考慮した波高確率密度分布の構築。

    レイリー分布をベースに砕波限界による切断を適用する。

    Args:
        x1: 砕波限界上限 (H/H0)
        x2: 砕波限界下限 (H/H0)
        aks: 浅水係数
        j: ガウス求積点のインデックス
        pxi: 求積点の確率重み
        xp: 波高比の離散点リスト
        p: 波高確率密度のリスト（in-place 更新）

    Returns:
        (xp, p): 更新された離散点と確率密度
    """
    mp = MP

    if j == 0:
        xmax = np.max([x1, 5.0])
        dx = xmax / float(mp - 1)
        xp = np.arange(0, xmax + dx, dx)

    a = 1.416 / aks
    for k in range(0, mp):
        p0 = 2.0 * a * a * xp[k] * np.exp(-((a * xp[k]) ** 2))
        p01 = 2.0 * a * a * x1 * np.exp(-((a * x1) ** 2))

        if xp[k] <= x2:
            p[k] = p[k] + pxi * p0
        elif xp[k] >= x1:
            pass
        else:
            p[k] = p[k] + pxi * (p0 - (xp[k] - x2) / (x1 - x2) * p01)
    return xp, p


def prob01(p: list[float], xp: list[float]) -> list[float]:
    """確率密度分布の正規化。

    確率密度の積分値が 1 になるように正規化する。

    Args:
        p: 波高確率密度のリスト（in-place 更新）
        xp: 波高比の離散点リスト

    Returns:
        正規化された確率密度リスト
    """
    mp = len(p)
    psum = 0.0
    for i in range(mp - 1):
        psum = psum + 0.5 * (p[i] + p[i + 1]) * (xp[i + 1] - xp[i])
    if psum == 0:
        return p
    for i in range(mp):
        p[i] = p[i] / psum
    return p


def shoal(dl0: float, h0l0: float) -> float:
    """合田モデルによる浅水変形係数の計算。

    浅水域における波高変化を表す浅水係数を、3つの領域に分けて計算する:
      - dl0 >= d30l0: 微小振幅波理論による浅水係数
      - d50l0 < dl0 < d30l0: べき乗則による遷移領域
      - dl0 <= d50l0: 砕波後のエネルギー散逸領域

    Args:
        dl0: 水深沖波波長比 (d/L0)
        h0l0: 沖波波形勾配 (H0/L0)

    Returns:
        浅水係数 Ks
    """
    if dl0 <= 0:
        raise ValueError(f"dl0 は正の値でなければなりません: {dl0}")
    if h0l0 <= 0:
        raise ValueError(f"h0l0 は正の値でなければなりません: {h0l0}")

    alfa = pi2 / 30.0 * h0l0
    dl = 0.1
    dl1 = 1.0

    while np.abs((dl - dl1) / dl) > CONVERGENCE_TOL:
        dl = dl1
        sh4 = np.sinh(2.0 * pi2 * dl)
        th2 = np.tanh(pi2 * dl)
        ch2 = np.cosh(pi2 * dl)
        f = (dl * th2) ** 2 - alfa / np.sqrt((1.0 + 2.0 * pi2 * dl / sh4) * th2)

        df = (
            2.0 * dl * th2**2
            + 2.0 * dl**2 * pi2 / ch2**2
            + pi2
            * alfa
            * np.sqrt(sh4)
            * th2
            * (2.0 - pi2 * dl * th2)
            / ((sh4 + 2.0 * pi2 * dl) * th2) ** 1.5
        )
        dl1 = dl - f / df

    d30l0 = dl * th2

    if dl0 >= d30l0:
        return aksi(dl0)

    aksi30 = aksi(d30l0)
    beta = pi2 / 50.0 * h0l0 * aksi30 * d30l0 ** (2.0 / 7.0)
    dl = 0.1
    dl1 = 99
    while abs((dl - dl1) / dl) > CONVERGENCE_TOL:
        dl = dl1
        f = dl * dl - beta * dl ** (-2.0 / 7.0)
        df = 2.0 * dl + 2.0 / 7.0 * beta * dl ** (-9.0 / 7.0)
        dl1 = dl - f / df
    dl = dl1
    d50l0 = dl1

    if dl0 > d50l0:
        return aksi30 * (d30l0 / dl0) ** (2.0 / 7.0)

    aks50 = aksi30 * (d30l0 / d50l0) ** (2.0 / 7.0)
    c50 = aks50 * d50l0**1.5 * (
        np.sqrt(pi2 * h0l0 * aks50) - 2.0 * np.sqrt(3.0) * d50l0
    )

    bb = 2.0 * np.sqrt(3.0) / np.sqrt(pi2 * h0l0) * dl0
    cc = c50 / np.sqrt(pi2 * h0l0) * dl0 ** (-1.5)
    a = aks50
    a1 = 99

    while np.abs((a - a1) / a) > CONVERGENCE_TOL:
        a = a1
        f = a * (np.sqrt(a) - bb) - cc
        df = 1.5 * np.sqrt(a) - bb
        a1 = a - f / df

    return a1


def cal_surf_goda(
    tant: float, h0l0: float, dl0: list[float] | None = None
) -> pd.DataFrame:
    """合田の砕波変形モデル。

    指定された海底勾配と波形勾配に対して、各水深での代表波高比を計算する。

    Args:
        tant: 海底勾配 (tan θ)
        h0l0: 沖波波形勾配 (H0/L0)
        dl0: 水深波長比 (d/L0) のリスト。None の場合はデフォルト値を使用。

    Returns:
        各水深における代表波高比等を含む DataFrame。
        カラム: H1_1000, H1_250, H1_100, H1_50, H1_10, H1_5, H1_3,
               dh0, dl0, etal0, aks
    """
    if tant < 0:
        raise ValueError(f"海底勾配は非負でなければなりません: {tant}")
    if h0l0 <= 0:
        raise ValueError(f"波形勾配は正の値でなければなりません: {h0l0}")

    if dl0 is None:
        dl0 = goda_dl0_list

    iflag = [1] * len(dl0)
    ipoint = len(dl0)
    etl0 = [0.0] * ipoint
    h2l0 = [0.0] * ipoint

    hnh0 = []
    dh0 = []
    xp_ = []
    p_ = []
    aks_ = []

    for i in range(ipoint):
        aks = shoal(dl0[i], h0l0)

        etl0 = etai(i, dl0, etl0)

        p = [0.0] * MP
        xp = [0.0] * MP

        for j in range(len(GAUSS_HERMITE_ABSCISSAE)):
            xil0, pxi = sbeat(h0l0, dl0[i], j)

            ddl0 = dl0[i] + etl0[i] + xil0
            ddl0 = max(ddl0, 0)
            x1, x2 = bindx(ddl0, h0l0, tant)
            xp, p = prob(x1, x2, aks, j, pxi, xp, p)

        p = prob01(p, xp)

        etl0, h2l0 = setup(etl0, dl0, h0l0, p, xp, i, h2l0)
        dh0_val, hnh_val = out(h0l0, dl0, etl0, p, xp, iflag, i)

        hnh0.append(hnh_val)
        dh0.append(dh0_val)
        xp_.append(xp)
        p_.append(p)
        aks_.append(aks)

    df = pd.DataFrame(
        hnh0, columns=["H1_1000", "H1_250", "H1_100", "H1_50", "H1_10", "H1_5", "H1_3"]
    )
    df["dh0"] = dh0
    df["dl0"] = dl0
    df["etal0"] = etl0
    df["aks"] = aks_
    return df


def cal_surf_goda_point(tant: float, h0l0: float, dl0: float) -> pd.DataFrame:
    """単一水深での合田の砕波変形モデル計算。

    Args:
        tant: 海底勾配 (tan θ)
        h0l0: 沖波波形勾配 (H0/L0)
        dl0: 水深波長比 (d/L0)

    Returns:
        代表波高比等を含む DataFrame（1行）
    """
    return cal_surf_goda(tant, h0l0, [dl0])


def cal_surf_goda_dim(
    tant: float,
    H0: float,
    T: float,
    dim: bool = False,
    d: float | list[float] | None = None,
) -> pd.DataFrame:
    """有次元入力による合田の砕波変形モデル計算。

    Args:
        tant: 海底勾配 (tan θ)
        H0: 沖波波高 [m]
        T: 周期 [s]
        dim: True の場合、有次元の結果カラムを追加
        d: 水深 [m]（単一値、リスト、または None）

    Returns:
        各水深における代表波高比等を含む DataFrame。
        dim=True の場合は有次元カラム (d, H1_3_dim 等) も追加。
    """
    if H0 <= 0:
        raise ValueError(f"沖波波高は正の値でなければなりません: {H0}")
    if T <= 0:
        raise ValueError(f"周期は正の値でなければなりません: {T}")

    L0 = g * T**2 / pi2
    h0l0 = H0 / L0

    if d is None:
        dl0: list[float] | None = None
    elif isinstance(d, (int, float)):
        dl0 = [d / L0]
    else:
        dl0 = [di / L0 for di in d]

    df = cal_surf_goda(tant, h0l0, dl0)

    if dim:
        df["d"] = df["dl0"] * L0
        for col in ["H1_3", "H1_250", "H1_100", "H1_50", "H1_10", "H1_5"]:
            df[f"{col}_dim"] = df[col] * H0

    return df
