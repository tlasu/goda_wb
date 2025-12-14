import numpy as np
import pandas as pd
from goda_wb.constant import pi2, g, goda_dl0_list


def aksi(dl0: float) -> float:
    """
    微小振幅波理論に基づく浅水係数の算出
    dl0:水深沖波波長比
    """

    dl = wave(pi2 * dl0) / pi2
    sh4 = np.sinh(2.0 * pi2 * dl)
    th2 = np.tanh(pi2 * dl)
    ksi = 1.0 / np.sqrt((1.0 + 2.0 * pi2 * dl / sh4) * th2)
    return ksi


def cal_wave_length(d: float, T: float) -> float:
    """
    波長の算出
    d:水深
    T:周期
    return:
        波長
    """
    dl0 = d / (g * T**2 / pi2)
    return pi2 * d / wave(pi2 * dl0)


def wave(d: float) -> float:
    """
    微小振幅波の波長
    d:2pi*d/L0
    return:
        2pi*d/L
    """
    if d > 10.0:
        xx = d
    else:
        if d < 1.0:
            x = np.sqrt(d)
        else:
            x = d
        e = 99.9
        while np.abs(e) > 0.0003:
            cothx = 1.0 / np.tanh(x)
            xx = x - (x - d * cothx) / (1.0 + d * (cothx**2 - 1.0))
            e = 1.0 - xx / x
            x = xx
    wave = xx
    return wave


def etai(i, dl0, etl0):
    if i <= 1:
        etl0[i] = 0.0
        return etl0
    else:
        etl0[i] = etl0[i - 1] + (etl0[i - 1] - etl0[i - 2]) * (dl0[i] - dl0[i - 1]) / (
            dl0[i - 1] - dl0[i - 2]
        )
        return etl0


def sbeat(h0l0: float, dl0: float, j: int) -> tuple[float, float]:
    rat = [3.2831, 2.3158, 1.3832, 0.4599, -0.4599, -1.3832, -2.3158, -3.2831]
    ppp = [0.0014, 0.0214, 0.1359, 0.3413, 0.3413, 0.1359, 0.0214, 0.0014]
    xil0 = rat[j] * 0.01 * h0l0 / np.sqrt(h0l0 * (1.0 + dl0 / h0l0))
    pxi = ppp[j]
    return xil0, pxi


def bindx(ddl0: float, h0l0: float, tant: float) -> tuple[float, float]:
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
    # mmax=100
    # mmax=len(p)
    mp = len(p)
    # dimension dl0(mmax),etl0(mmax),h2l0(mmax)
    # dimension p(mp),xp(mp)

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
    mmax = 100
    mp = 100
    mmax = 51
    mp = 51
    nwave = 7
    hnh0 = [0.0 for i in range(nwave)]
    hnd = [0.0 for i in range(nwave)]

    epn = [1 / 1000, 0.004, 0.08333, 0.02, 0.1, 0.2, 0.3333333]
    # 代表波高: 1/1000, 1/250, 1/120, 1/50, 1/10, 1/5, 1/3

    xxx = dl0[i] / h0l0

    # calculate representative wave height
    # hbar, hrms

    # Convert to NumPy arrays for vectorization
    xp_arr = np.asarray(xp)
    p_arr = np.asarray(p)

    # Vectorized computation using array slicing
    dx = xp_arr[1:] - xp_arr[:-1]
    p_k = p_arr[:-1]
    p_k1 = p_arr[1:]
    xp_k = xp_arr[:-1]
    xp_k1 = xp_arr[1:]

    # Vectorized xb calculation
    xb = np.sum(dx / 6.0 * ((2.0 * p_k + p_k1) * xp_k + (p_k + 2.0 * p_k1) * xp_k1))

    # Vectorized xxb calculation
    xxb = np.sum(
        dx
        / 12.0
        * (
            (3.0 * p_k + p_k1) * xp_k**2
            + 2.0 * (p_k + p_k1) * xp_k * xp_k1
            + (p_k + 3.0 * p_k1) * xp_k1**2
        )
    )
    hbarh0 = xb
    hbard = hbarh0 * h0l0 / dl0[i]

    hrmsh0 = np.sqrt(xxb)
    hrmsd = hrmsh0 * h0l0 / dl0[i]

    # print('representative wave height')
    # print('hbar/h0 =',hbarh0)
    # print('hrms/h0 =',hrmsh0)

    #   calculate representative wave height
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

    for n in range(nwave):
        La = 0
        for k in range(mp - 2, -1, -1):
            if (ep[k + 1] < epn[n]) and (ep[k] >= epn[n]):
                La = 1
                k1 = k + 1
                k2 = k
                ratk = (epn[n] - ep[k + 1]) / (ep[k] - ep[k + 1])
                break
        if La == 0:
            hnh0[n] = 0.0
        else:
            hnh0[n] = (1.0 - ratk) * hep[k1] + ratk * hep[k2]
        hnd[n] = hnh0[n] * h0l0 / dl0[i]

    # yy1=hnh0[5]
    # yy2=hnh0[0]
    # yy3=hnh0[1]
    return xxx, hnh0


def prob(
    x1: float,
    x2: float,
    aks: float,
    j: int,
    pxi: float,
    xp: list[float],
    p: list[float],
) -> tuple[list[float], list[float]]:
    mp = 51

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
            p[k] = p[k]
        else:
            p[k] = p[k] + pxi * (p0 - (xp[k] - x2) / (x1 - x2) * p01)
    return xp, p


def prob01(p, xp):
    mp = len(p)
    psum = 0.0
    for i in range(mp - 1):
        psum = psum + 0.5 * (p[i] + p[i + 1]) * (xp[i + 1] - xp[i])
    for i in range(mp):
        p[i] = p[i] / psum
    return p


class cla_shoal:
    def __init__(self, dl0: float, h0l0: float):
        self.dl0 = dl0
        self.h0l0 = h0l0
        self.alfa = pi2 / 30.0 * h0l0
        self.dl = 0.1
        self.dl1 = 1.0
        self.n = 0
        self.d30l0 = 0.0
        self.d50l0 = 0.0
        self.aks = 0.0
        self.aks30 = 0.0
        self.aks50 = 0.0

    def calc(self):
        #
        #  find out h30
        #
        while np.abs((self.dl - self.dl1) / self.dl) > 1.0e-6:
            self.dl = self.dl1
            self.n = self.n + 1
            sh4 = np.sinh(2.0 * pi2 * self.dl)
            th2 = np.tanh(pi2 * self.dl)
            ch2 = np.cosh(pi2 * self.dl)
            f = (self.dl * th2) ** 2 - self.alfa / np.sqrt(
                (1.0 + 2.0 * pi2 * self.dl / sh4) * th2
            )
            df = (
                2.0 * self.dl * th2**2
                + 2.0 * self.dl**2 * pi2 / ch2**2
                + pi2
                * self.alfa
                * np.sqrt(sh4)
                * th2
                * (2.0 - pi2 * self.dl * th2)
                / ((sh4 + 2.0 * pi2 * self.dl) * th2) ** 1.5
            )
            self.dl1 = self.dl - f / df

        self.d30l0 = self.dl * th2
        #
        #  find out h50
        #
        aksi30 = aksi(self.d30l0)
        beta = pi2 / 50.0 * self.h0l0 * aksi30 * self.d30l0 ** (2.0 / 7.0)

        self.dl = 0.1
        self.dl1 = 99
        while np.abs((self.dl - self.dl1) / self.dl) > 1.0e-6:
            self.dl = self.dl1
            f = self.dl * self.dl - beta * self.dl ** (-2.0 / 7.0)
            df = 2.0 * self.dl + 2.0 / 7.0 * beta * self.dl ** (-9.0 / 7.0)
            self.dl1 = self.dl - f / df

        self.dl = self.dl1
        self.d50l0 = self.dl1


def shoal(dl0: float, h0l0: float) -> float:
    """
    浅水変形の計算
    dl0:水深沖波波長比
    h0l0:沖波波形勾配
    """

    alfa = pi2 / 30.0 * h0l0
    dl = 0.1
    dl1 = 1.0
    n = 0

    while np.abs((dl - dl1) / dl) > 1.0e-6:
        dl = dl1
        n = n + 1
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

    #  ks for  h > h30
    if dl0 >= d30l0:
        aks = aksi(dl0)
        return aks

    #
    #  find out h50
    #

    aksi30 = aksi(d30l0)
    beta = pi2 / 50.0 * h0l0 * aksi30 * d30l0 ** (2.0 / 7.0)
    dl = 0.1
    dl1 = 99
    while abs((dl - dl1) / dl) > 1.0e-6:
        dl = dl1
        f = dl * dl - beta * dl ** (-2.0 / 7.0)
        df = 2.0 * dl + 2.0 / 7.0 * beta * dl ** (-9.0 / 7.0)
        dl1 = dl - f / df
    dl = dl1
    d50l0 = dl1

    #
    #  ks for h50< h < h30
    #
    if dl0 > d50l0:
        aks = aksi30 * (d30l0 / dl0) ** (2.0 / 7.0)
        return aks
    #
    #  ks for h50 < h
    #
    aks50 = aksi30 * (d30l0 / d50l0) ** (2.0 / 7.0)
    c50 = (
        aks50 * d50l0**1.5 * (np.sqrt(pi2 * h0l0 * aks50) - 2.0 * np.sqrt(3.0) * d50l0)
    )

    bb = 2.0 * np.sqrt(3.0) / np.sqrt(pi2 * h0l0) * dl0
    cc = c50 / np.sqrt(pi2 * h0l0) * dl0 ** (-1.5)
    a = aks50
    a1 = 99

    while np.abs((a - a1) / a) > 1.0e-6:
        a = a1
        f = a * (np.sqrt(a) - bb) - cc
        df = 1.5 * np.sqrt(a) - bb
        a1 = a - f / df

    aks = a1
    return aks


def cal_surf_goda(
    tant: float, h0l0: float, dl0: list[float] | float = None
) -> pd.DataFrame:
    """
    合田の砕波変形モデル
    h0l0:波形勾配
    tant:海底勾配
    dl0:水深波長比
    """
    assert isinstance(dl0, list) or dl0 is None

    if dl0 is None:
        dl0 = goda_dl0_list
    if isinstance(dl0, float):
        dl0 = [dl0]

    iflag = [1 for i in np.arange(len(dl0))]
    ipoint = len(dl0)
    etl0 = [0.0 for i in range(ipoint)]
    h2l0 = [0.0 for i in range(ipoint)]

    hnh0 = []
    dh0 = []
    xp_ = []
    p_ = []
    aks_ = []

    for m in range(0, ipoint):
        i = m
        xxx = dl0[i]
        aks = shoal(xxx, h0l0)  # 浅水係数

        etl0 = etai(i, dl0, etl0)  # etal0の初期化

        p = [0.0 for i in range(51)]
        xp = [0.0 for i in range(51)]

        for n in range(0, 8):
            j = n
            xxx = dl0[i]
            xil0, pxi = sbeat(h0l0, xxx, j)

            ddl0 = dl0[i] + etl0[i] + xil0
            ddl0 = max(ddl0, 0)
            x1, x2 = bindx(ddl0, h0l0, tant)
            xp, p = prob(x1, x2, aks, j, pxi, xp, p)

        p = prob01(p, xp)

        etl0, h2l0 = setup(etl0, dl0, h0l0, p, xp, i, h2l0)
        xxx_, hnh_ = out(h0l0, dl0, etl0, p, xp, iflag, i)

        hnh0.append(hnh_)
        dh0.append(xxx_)
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
    return cal_surf_goda(tant, h0l0, [dl0])


def cal_surf_goda_dim(
    tant: float,
    H0: float,
    T: float,
    dim: bool = False,
    d:float|list[float]|None=None,
) -> pd.DataFrame:
    """
    合田の砕波変形モデル
    H0:沖波波高
    T:周期
    d:対象水深
    """
    h0l0 = H0 / (g * T**2 / pi2)

    if d is None:
        dl0 = goda_dl0_list
    elif isinstance(d, float):
        dl0 = d/(g * T**2 / pi2)
    else:
        dl0 = [d[i]/(g * T**2 / pi2) for i in range(len(d))]
    df = cal_surf_goda(tant, h0l0, dl0)
    if dim:
        df["d"] = df["dl0"] * (g * T**2 / pi2)
        df["H1_3_dim"] = df["H1_3"] * H0
        df["H1_250_dim"] = df["H1_250"] * H0
        df["H1_100_dim"] = df["H1_100"] * H0
        df["H1_50_dim"] = df["H1_50"] * H0
        df["H1_10_dim"] = df["H1_10"] * H0
        df["H1_5_dim"] = df["H1_5"] * H0
        df["H1_3_dim"] = df["H1_3"] * H0
        return df
    else:
        return df
