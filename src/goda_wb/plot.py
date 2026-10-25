"""合田 (1975) の算定図（原図）に計算結果を重ねて描く。

原図: 合田良実 (1975) 浅海域における波浪の砕波変形, 港湾技術研究所報告 14(3)
  - p.96 図-50〜53 砕波帯内の有義波高の算定図 (H1/3)
  - p.97 図-54〜57 砕波帯内の最高波高の算定図 (Hmax = H1/250)
  - p.99 図-61 砕波帯内の有義波高の略算式と計算値の比較 (H0'/L0 = 0.02)
  - p.100 図-62 砕波帯内の最高波高の略算式と計算値の比較 (H0'/L0 = 0.04)

スキャン画像 (400 dpi) 上の各図の枠の四隅の画素座標を読み取り、双一次補間で
(h/H0', H/H0') を画素座標へ変換して描画する。ただし図-52 は図枠の下辺が方眼と
ずれて描かれているため、方眼を基準に較正している。

matplotlib が必要（``uv add "goda-wb[plot]"``）。
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

import goda_wb.core as core

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# 算定図の横軸 h/H0' の上限
CHART_DH0_MAX = 4.0

# 算定図がある海底勾配。図-50〜53, 54〜57 の並び順に対応する。
CHART_SLOPES: tuple[float, ...] = (1 / 10, 1 / 20, 1 / 30, 1 / 100)

# 図枠の四隅 (x0=0, x1=4.0, y0=0, y1=ymax) の画素座標 (x, y)
# LB: 左下, RB: 右下, LT: 左上, RT: 右上。並びは CHART_SLOPES と同じ。
GODA1975_CHARTS: dict[str, dict[str, Any]] = {
    "H1_3": {
        "image": "goda1975_p96_fig50-53.png",
        "ymax": 3.0,
        "approx": "H1_3",
        "figs": (50, 51, 52, 53),
        "corners": (
            {
                "LB": (362.5, 1809.0),
                "RB": (1287.0, 1810.5),
                "LT": (364.0, 432.5),
                "RT": (1289.5, 430.5),
            },
            {
                "LB": (1496.5, 1810.0),
                "RB": (2419.0, 1816.0),
                "LT": (1503.5, 435.0),
                "RT": (2430.5, 441.5),
            },
            # 図-52 は図枠の下辺が方眼より 0.033 (左)〜0.056 (右) 高く描かれているため、
            # 方眼 (0.1 間隔 45.7 px) から外挿した y=0 の位置を下側の隅とする
            {
                "LB": (338.4, 3421.6),
                "RB": (1261.8, 3433.6),
                "LT": (350.0, 2052.5),
                "RT": (1273.0, 2061.5),
            },
            {
                "LB": (1487.0, 3418.5),
                "RB": (2405.0, 3424.5),
                "LT": (1495.5, 2056.0),
                "RT": (2410.5, 2063.0),
            },
        ),
    },
    "H1_250": {
        "image": "goda1975_p97_fig54-57.png",
        "ymax": 3.5,
        "approx": "Hmax",
        "figs": (54, 55, 56, 57),
        "corners": (
            {
                "LB": (545.0, 1861.5),
                "RB": (1335.5, 1864.5),
                "LT": (552.5, 490.5),
                "RT": (1344.5, 495.5),
            },
            {
                "LB": (1688.5, 1869.0),
                "RB": (2490.5, 1875.5),
                "LT": (1700.0, 489.5),
                "RT": (2501.0, 496.0),
            },
            {
                "LB": (521.0, 3459.5),
                "RB": (1319.0, 3465.5),
                "LT": (533.0, 2081.5),
                "RT": (1330.5, 2088.5),
            },
            {
                "LB": (1693.5, 3464.5),
                "RB": (2484.0, 3466.5),
                "LT": (1700.5, 2103.0),
                "RT": (2490.0, 2104.0),
            },
        ),
    },
}

# 図-61・62: 原図の記号が合田の計算値、線が略算式。
GODA1975_APPROX_FIGS: dict[int, dict[str, Any]] = {
    61: {
        "image": "goda1975_p99_fig61.png",
        "h0l0": 0.02,
        "column": "H1_3",
        "approx": "H1_3",
        "ymax": 1.5,
        "corners": {
            "LB": (213.0, 727.0),
            "RB": (1143.0, 726.0),
            "LT": (211.0, 42.0),
            "RT": (1141.0, 39.5),
        },
    },
    62: {
        "image": "goda1975_p100_fig62.png",
        "h0l0": 0.04,
        "column": "H1_250",
        "approx": "Hmax",
        "ymax": 2.5,
        "corners": {
            "LB": (110.5, 1203.0),
            "RB": (1054.0, 1204.5),
            "LT": (115.0, 45.5),
            "RT": (1055.0, 44.5),
        },
    },
}

# 原図の枠の外にある目盛・軸ラベルを表示するための余白 [px] (左, 右, 下, 上)
CHART_PADDING = (150, 40, 110, 40)

# 曲線を描く h/H0' の刻み
CHART_DH0_STEP = 0.02


def _pyplot():
    """matplotlib.pyplot を import して返す。未インストールなら案内付きで ImportError。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - 環境依存
        raise ImportError('goda_wb.plot には matplotlib が必要です: uv add "goda-wb[plot]"') from e
    return plt


def _normalize_column(column: str) -> str:
    """列名を算定図のキー (H1_3 / H1_250) に揃える。Hmax は H1_250 とみなす。"""
    key = "H1_250" if column == "Hmax" else column
    if key not in GODA1975_CHARTS:
        raise ValueError(f"column は H1_3, H1_250 (Hmax) のいずれかです: {column}")
    return key


def chart_index(tant: float) -> int:
    """海底勾配に対応する算定図の番号 (CHART_SLOPES の添字) を返す。

    Args:
        tant: 海底勾配 tanθ。1/10, 1/20, 1/30, 1/100 のいずれか。

    Returns:
        int: CHART_SLOPES での添字。

    Raises:
        ValueError: 算定図のない勾配の場合。
    """
    for i, slope in enumerate(CHART_SLOPES):
        if np.isclose(tant, slope, rtol=1e-6):
            return i
    raise ValueError(f"算定図があるのは勾配 1/10, 1/20, 1/30, 1/100 のみです: {tant}")


def chart_to_pixel(
    x: float | Sequence[float] | np.ndarray,
    y: float | Sequence[float] | np.ndarray,
    corners: dict[str, tuple[float, float]],
    ymax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """図の座標 (h/H0', H/H0') を双一次補間で画素座標へ変換する。

    Args:
        x: 横軸の値 h/H0'（図枠の左端 0、右端 CHART_DH0_MAX）。
        y: 縦軸の値 H/H0'（図枠の下端 0、上端 ymax）。
        corners: 図枠の四隅の画素座標 {"LB", "RB", "LT", "RT"}。
        ymax: 図枠の上端の縦軸の値。

    Returns:
        tuple[np.ndarray, np.ndarray]: 画素座標 (px, py)。
    """
    u = np.atleast_1d(np.asarray(x, dtype=float)) / CHART_DH0_MAX
    v = np.atleast_1d(np.asarray(y, dtype=float)) / ymax
    lb, rb, lt, rt = (np.array(corners[k]) for k in ("LB", "RB", "LT", "RT"))
    p = (
        np.outer((1 - u) * (1 - v), lb)
        + np.outer(u * (1 - v), rb)
        + np.outer((1 - u) * v, lt)
        + np.outer(u * v, rt)
    )
    return p[:, 0], p[:, 1]


def load_chart_image(name: str) -> np.ndarray:
    """同梱の原図スキャン画像を読み込む。

    Args:
        name: 画像ファイル名（GODA1975_CHARTS などの "image"）。

    Returns:
        np.ndarray: 画像データ。
    """
    import matplotlib.image as mpimg

    with resources.as_file(resources.files("goda_wb") / "data" / "goda1975" / name) as path:
        return mpimg.imread(path)


def _show_image(
    ax: Axes, image: str, corners: dict[str, tuple[float, float]], alpha: float
) -> None:
    """原図を表示し、図枠と目盛・軸ラベルが収まる範囲に切り出す。"""
    ax.imshow(load_chart_image(image), cmap="gray", alpha=alpha)
    xs = [c[0] for c in corners.values()]
    ys = [c[1] for c in corners.values()]
    left, right, bottom, top = CHART_PADDING
    ax.set_xlim(min(xs) - left, max(xs) + right)
    ax.set_ylim(max(ys) + bottom, min(ys) - top)
    ax.set_axis_off()


def draw_goda1975_chart(
    tant: float, column: str = "H1_3", ax: Axes | None = None, alpha: float = 0.6
) -> Axes:
    """合田 (1975) の算定図（図-50〜57）を描く。

    Args:
        tant: 海底勾配 tanθ（1/10, 1/20, 1/30, 1/100）。
        column: "H1_3"（図-50〜53）または "H1_250" / "Hmax"（図-54〜57）。
        ax: 描画先。None なら新しい図を作る。
        alpha: 原図の不透明度。

    Returns:
        Axes: 原図を描いた Axes。
    """
    key = _normalize_column(column)
    k = chart_index(tant)
    chart = GODA1975_CHARTS[key]
    if ax is None:
        _, ax = _pyplot().subplots(figsize=(6, 7.5))
    _show_image(ax, chart["image"], chart["corners"][k], alpha)
    ax.set_title(f"Goda (1975) Fig. {chart['figs'][k]}  slope 1/{round(1 / tant)}  {key}/H0'")
    return ax


def plot_on_goda1975(
    tant: float,
    h0l0: float | Sequence[float],
    column: str = "H1_3",
    *,
    ax: Axes | None = None,
    approx: bool = True,
    points: Sequence[tuple[float, float]] | None = None,
    colors: Sequence[Any] | None = None,
    alpha: float = 0.6,
) -> Axes:
    """合田 (1975) の算定図（図-50〜57）に計算結果を重ねて描く。

    原図の上に cal_surf_goda の結果を実線で、approx=True なら
    cal_surf_goda_approx（略算式）の結果を破線で描く。

    Args:
        tant: 海底勾配 tanθ（1/10, 1/20, 1/30, 1/100）。
        h0l0: 波形勾配 H0/L0。複数指定すると曲線を複数描く。
        column: "H1_3"（図-50〜53）または "H1_250" / "Hmax"（図-54〜57）。
        ax: 描画先。None なら新しい図に原図を描く。原図がまだない Axes には原図も描く。
        approx: True なら略算式の曲線も描く。
        points: 図に打つ点 (h/H0', H/H0') のリスト。
        colors: 各 h0l0 の曲線の色。None なら matplotlib の既定の色順。
        alpha: 原図の不透明度。

    Returns:
        Axes: 描画した Axes。

    Raises:
        ValueError: 算定図のない勾配や列名の場合。
    """
    key = _normalize_column(column)
    k = chart_index(tant)
    chart = GODA1975_CHARTS[key]
    corners, ymax = chart["corners"][k], chart["ymax"]
    if ax is None or not ax.images:
        ax = draw_goda1975_chart(tant, key, ax=ax, alpha=alpha)

    h0l0_list = [h0l0] if np.isscalar(h0l0) else list(h0l0)
    if colors is None:
        colors = [f"C{n % 10}" for n in range(len(h0l0_list))]
    dh0 = np.arange(CHART_DH0_MAX, 0.0, -CHART_DH0_STEP)
    for h, color in zip(h0l0_list, colors, strict=True):
        dl0 = list(dh0 * h)
        df = core.cal_surf_goda(tant, h, dl0)
        x = df["dh0"].to_numpy()
        y = df[key].to_numpy()
        inside = y <= ymax
        px, py = chart_to_pixel(x[inside], y[inside], corners, ymax)
        ax.plot(px, py, "-", color=color, linewidth=2.5, alpha=0.8, label=f"H0'/L0={h:g}")
        if approx:
            ya = core.cal_surf_goda_approx(tant, h, dl0)[chart["approx"]].to_numpy()
            inside = ya <= ymax
            px, py = chart_to_pixel(x[inside], ya[inside], corners, ymax)
            ax.plot(px, py, "--", color=color, linewidth=1.8, alpha=0.9)

    if points:
        xs, ys = zip(*points, strict=True)
        px, py = chart_to_pixel(xs, ys, corners, ymax)
        ax.plot(px, py, "o", ms=9, color="tab:red", mec="white", mew=2, zorder=5)
    return ax


def plot_goda1975_approx_fig(
    fig_no: int,
    tants: Sequence[float] = (1 / 10, 1 / 100),
    *,
    ax: Axes | None = None,
    colors: Sequence[Any] = ("tab:red", "tab:blue"),
    alpha: float = 0.6,
) -> Axes:
    """合田 (1975) 図-61・62（略算式と計算値の比較）に計算結果を重ねて描く。

    原図の記号が合田の計算値、線が略算式。これに cal_surf_goda の結果を実線で、
    cal_surf_goda_approx の結果を破線で重ねる。

    Args:
        fig_no: 図番号 61（有義波高, H0'/L0=0.02）または 62（最高波高, H0'/L0=0.04）。
        tants: 描く海底勾配 tanθ のリスト。
        ax: 描画先。None なら新しい図を作る。
        colors: 各勾配の曲線の色。
        alpha: 原図の不透明度。

    Returns:
        Axes: 描画した Axes。

    Raises:
        ValueError: fig_no が 61, 62 以外の場合。
    """
    if fig_no not in GODA1975_APPROX_FIGS:
        raise ValueError(f"fig_no は 61 または 62 です: {fig_no}")
    spec = GODA1975_APPROX_FIGS[fig_no]
    if ax is None:
        _, ax = _pyplot().subplots(figsize=(7, 6))
    ax.imshow(load_chart_image(spec["image"]), cmap="gray", alpha=alpha)
    ax.set_axis_off()

    dh0 = np.arange(CHART_DH0_MAX, 0.0, -CHART_DH0_STEP)
    dl0 = list(dh0 * spec["h0l0"])
    for tant, color in zip(tants, colors, strict=False):
        df = core.cal_surf_goda(tant, spec["h0l0"], dl0)
        da = core.cal_surf_goda_approx(tant, spec["h0l0"], dl0)
        label = f"1/{round(1 / tant)}"
        for y, style, lw, kind in [
            (df[spec["column"]], "-", 2.5, "computed"),
            (da[spec["approx"]], "--", 1.8, "approx"),
        ]:
            px, py = chart_to_pixel(df["dh0"], y, spec["corners"], spec["ymax"])
            ax.plot(px, py, style, color=color, linewidth=lw, alpha=0.8, label=f"{label} {kind}")
    ax.set_title(f"Goda (1975) Fig. {fig_no}  H0'/L0={spec['h0l0']}")
    return ax
