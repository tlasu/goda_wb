"""合田 (1975) の算定図（原図）に本ライブラリの計算結果を重ねて比較する。

原図: 合田良実 (1975) 浅海域における波浪の砕波変形, 港湾技術研究所報告 14(3)
  - p.96 図-50〜53 砕波帯内の有義波高の算定図 (H1/3)
  - p.97 図-54〜57 砕波帯内の最高波高の算定図 (Hmax = H1/250)

スキャン画像 (400 dpi) 上の各図の枠の四隅の画素座標を読み取り、双一次補間で
(h/H0', H/H0') を画素座標へ変換して描画する。

    cd sample && uv run python compare_goda1975.py
"""

# %%
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import goda_wb.core as core

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "goda1975"

LIST_H0L0 = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
SLOPES = [1 / 10, 1 / 20, 1 / 30, 1 / 100]
SLOPE_LABELS = ["1/10", "1/20", "1/30", "1/100"]

# 図枠の四隅 (x0=0, x1=4.0, y0=0, y1=ymax) の画素座標 (x, y)
# LB: 左下, RB: 右下, LT: 左上, RT: 右上。並びは SLOPES と同じ。
CHARTS = {
    "H1_3": {
        "image": "goda1975_p96_fig50-53.png",
        "ymax": 3.0,
        "ylabel": "$H_{1/3}/H_0'$",
        "figs": ["図-50", "図-51", "図-52", "図-53"],
        "corners": [
            {"LB": (362.5, 1809.0), "RB": (1287.0, 1810.5), "LT": (364.0, 432.5), "RT": (1289.5, 430.5)},
            {"LB": (1496.5, 1810.0), "RB": (2419.0, 1816.0), "LT": (1503.5, 435.0), "RT": (2430.5, 441.5)},
            {"LB": (338.5, 3406.5), "RB": (1262.0, 3408.0), "LT": (350.0, 2052.5), "RT": (1273.0, 2061.5)},
            {"LB": (1487.0, 3418.5), "RB": (2405.0, 3424.5), "LT": (1495.5, 2056.0), "RT": (2410.5, 2063.0)},
        ],
    },
    "H1_250": {
        "image": "goda1975_p97_fig54-57.png",
        "ymax": 3.5,
        "ylabel": "$H_{max}/H_0'$ ($H_{1/250}$)",
        "figs": ["図-54", "図-55", "図-56", "図-57"],
        "corners": [
            {"LB": (545.0, 1861.5), "RB": (1335.5, 1864.5), "LT": (552.5, 490.5), "RT": (1344.5, 495.5)},
            {"LB": (1688.5, 1869.0), "RB": (2490.5, 1875.5), "LT": (1700.0, 489.5), "RT": (2501.0, 496.0)},
            {"LB": (521.0, 3459.5), "RB": (1319.0, 3465.5), "LT": (533.0, 2081.5), "RT": (1330.5, 2088.5)},
            {"LB": (1693.5, 3464.5), "RB": (2484.0, 3466.5), "LT": (1700.5, 2103.0), "RT": (2490.0, 2104.0)},
        ],
    },
}


def to_pixel(x, y, corners, ymax):
    """図の座標 (h/H0', H/H0') を双一次補間で画素座標へ変換する。"""
    u = np.asarray(x) / 4.0
    v = np.asarray(y) / ymax
    lb, rb, lt, rt = (np.array(corners[k]) for k in ("LB", "RB", "LT", "RT"))
    p = (
        np.outer((1 - u) * (1 - v), lb)
        + np.outer(u * (1 - v), rb)
        + np.outer((1 - u) * v, lt)
        + np.outer(u * v, rt)
    )
    return p[:, 0], p[:, 1]


def compute(tant, column):
    dh0 = np.arange(4.0, 0.0, -0.05)
    curves = {}
    for h0l0 in LIST_H0L0:
        df = core.cal_surf_goda(tant, h0l0, list(dh0 * h0l0))
        curves[h0l0] = (df["dh0"].to_numpy(), df[column].to_numpy())
    return curves


def draw(column):
    chart = CHARTS[column]
    img = mpimg.imread(IMG_DIR / chart["image"])
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(12, 15))
    for k, ax in enumerate(axes.flat):
        corners = chart["corners"][k]
        xs = [c[0] for c in corners.values()]
        ys = [c[1] for c in corners.values()]
        pad = 40
        ax.imshow(img, cmap="gray", alpha=0.6)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)
        for n, (h0l0, (x, y)) in enumerate(compute(SLOPES[k], column).items()):
            inside = (x <= 4.0) & (y <= chart["ymax"])
            px, py = to_pixel(x[inside], y[inside], corners, chart["ymax"])
            ax.plot(px, py, "-", color=colors[n], linewidth=2.5, alpha=0.8,
                    label=f"$H_0'/L_0={h0l0}$")
        ax.set_title(f"{chart['figs'][k]} 海底勾配 {SLOPE_LABELS[k]}  {chart['ylabel']}")
        ax.set_axis_off()
    axes.flat[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=9)
    fig.suptitle("合田 (1975) 原図（灰色）と goda_wb の計算結果（色線）")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for column in CHARTS:
        fig = draw(column)
        fig.savefig(IMG_DIR / f"compare_{column}.png", dpi=80)
        plt.close(fig)
