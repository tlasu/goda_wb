"""合田 (2004) ECOH/YG 技術論文 No.2 の図に goda_wb.pegbis の計算結果を重ねて比較する。

原図: 合田良実 (2004) 不規則波浪変形計算における段階的砕波係数の導入について (Ver.5)
  - p.19 付図-4~7 一様勾配斜面上の Hmax と H1/3（記号が PEGBIS、線が合田 (1975) のモデル）
  - p.7 図-11 権による水平ステップ実験と計算値の比較

ページ画像は論文 PDF (https://www.ecoh.co.jp/goda/849/) から 300 dpi で作成する:

    pdftoppm -r 300 -png -gray -f 19 -l 19 合田論文_ECOHNo.2_ps.pdf sample/pegbis/ecoh2
    pdftoppm -r 300 -png -gray -f 7 -l 7 合田論文_ECOHNo.2_ps.pdf sample/pegbis/ecoh2
    cd sample && uv run python compare_pegbis_paper.py

各図の枠（付図）または方眼（図-11）の画素座標から双一次補間で図の座標を画素座標へ変換する。
"""

# %%
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np

import goda_wb.pegbis as pegbis

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "pegbis"

# 付図の枠の四隅 (x = 0, 4.0; y = 0, ymax) の画素座標 (x, y)
SLOPE_FIGS = [
    {"slope": 1 / 10, "title": "付図-4 海底勾配 1/10", "ymax": 2.5,
     "corners": {"LB": (377.0, 2908.0), "RB": (1111.5, 2908.0),
                 "LT": (377.0, 2119.5), "RT": (1111.5, 2119.5)}},
    {"slope": 1 / 20, "title": "付図-5 海底勾配 1/20", "ymax": 2.0,
     "corners": {"LB": (1416.0, 917.0), "RB": (2150.5, 917.0),
                 "LT": (1416.0, 287.5), "RT": (2150.5, 287.5)}},
    {"slope": 1 / 30, "title": "付図-6 海底勾配 1/30", "ymax": 2.0,
     "corners": {"LB": (1428.5, 1844.5), "RB": (2141.5, 1844.5),
                 "LT": (1428.5, 1235.5), "RT": (2141.5, 1235.5)}},
    {"slope": 1 / 100, "title": "付図-7 海底勾配 1/100", "ymax": 2.0,
     "corners": {"LB": (1428.5, 2757.5), "RB": (2141.5, 2757.5),
                 "LT": (1428.5, 2148.5), "RT": (2141.5, 2148.5)}},
]

# 図-11: x = -4, 4 m の縦軸と、波高 0 m, 0.20 m の方眼。水深は 0 m の線から下向きに 0.5 m が 126 px
FIG11 = {"corners": {"LB": (1416.0, 1892.0), "RB": (2122.5, 1892.0),
                     "LT": (1416.0, 1388.0), "RT": (2122.5, 1388.0)},
         "x": (-4.0, 4.0), "y": (0.0, 0.20), "depth_px_per_m": 252.0}


def to_pixel(x, y, corners, xr, yr):
    """図の座標を双一次補間で画素座標へ変換する。"""
    u = (np.asarray(x) - xr[0]) / (xr[1] - xr[0])
    v = (np.asarray(y) - yr[0]) / (yr[1] - yr[0])
    lb, rb, lt, rt = (np.array(corners[k]) for k in ("LB", "RB", "LT", "RT"))
    p = (
        np.outer((1 - u) * (1 - v), lb)
        + np.outer(u * (1 - v), rb)
        + np.outer((1 - u) * v, lt)
        + np.outer(u * v, rt)
    )
    return p[:, 0], p[:, 1]


def zoom(ax, corners, pad=60):
    xs = [c[0] for c in corners.values()]
    ys = [c[1] for c in corners.values()]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)
    ax.set_axis_off()


def draw_slopes():
    img = mpimg.imread(IMG_DIR / "ecoh2-19.png")
    dh0 = np.arange(4.0, 0.0, -0.05)
    styles = {0.02: "tab:red", 0.04: "tab:blue"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for spec, ax in zip(SLOPE_FIGS, axes.flat, strict=True):
        ax.imshow(img, cmap="gray", alpha=0.6)
        for h0l0, color in styles.items():
            res = pegbis.cal_surf_pegbis_slope(spec["slope"], h0l0, list(dh0 * h0l0), n_freq=10)
            for col, ls in [("H1_250", "-"), ("H1_3", "--")]:
                y = res[col].to_numpy()
                inside = y <= spec["ymax"]
                px, py = to_pixel(dh0[inside], y[inside], spec["corners"], (0, 4), (0, spec["ymax"]))
                name = "Hmax" if col == "H1_250" else "H1/3"
                ax.plot(px, py, ls, color=color, lw=2.0, alpha=0.8,
                        label=f"goda_wb {name}, $H_0/L_0$={h0l0}")
        zoom(ax, spec["corners"])
        ax.set_title(spec["title"])
    axes.flat[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("論文の付図-4~7（灰色：記号が PEGBIS、黒線が合田 (1975)）と goda_wb.pegbis（色線）")
    fig.tight_layout()
    return fig


def draw_fig11():
    img = mpimg.imread(IMG_DIR / "ecoh2-07.png")
    x = np.linspace(-4.0, 4.0, 161)
    h = np.interp(x, [-4.0, -3.0, -1.0, 0.0, 4.0], [0.36, 0.36, 0.16, 0.11, 0.11])
    # 図-11 は論文本文の図で Ver.3 以前の α_b = 0.125 で計算されている。M は論文の現地計算と同じ 301
    df = pegbis.cal_surf_pegbis(x, h, 0.0879, 2.21, n_freq=10, alpha_b=0.125, M=301)
    c = FIG11["corners"]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray", alpha=0.6)
    colors = plt.get_cmap("tab10").colors
    for n, (col, name) in enumerate(
        [("H1_250", "Hmax"), ("H1_10", "H1/10"), ("H1_3", "H1/3"), ("Hrms", "Hrms"),
         ("Hmean", "Hmean")]
    ):
        px, py = to_pixel(df["x"], df[col], c, FIG11["x"], FIG11["y"])
        ax.plot(px, py, color=colors[n], lw=2.0, alpha=0.8, label=f"goda_wb {name}")
    px, _ = to_pixel(x, np.zeros_like(x), c, FIG11["x"], FIG11["y"])
    ax.plot(px, c["LB"][1] + h * FIG11["depth_px_per_m"], color="tab:gray", lw=1.5,
            ls=":", label="入力した水深")
    zoom(ax, {**c, "BOT": (c["LB"][0], c["LB"][1] + 130)})
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("論文の図-11（灰色：線が論文の計算値、記号が実験値）と goda_wb.pegbis（色線）")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_slopes()
    fig.savefig(IMG_DIR / "overlay_slope.png", dpi=90)
    plt.close(fig)
    fig = draw_fig11()
    fig.savefig(IMG_DIR / "overlay_fig11.png", dpi=90)
    plt.close(fig)
