"""合田 (1975) の算定図（原図）に本ライブラリの計算結果と略算式を重ねて比較する。

重ね描きは goda_wb.plot の関数で行う（原図の画像と図枠の較正値はパッケージに同梱）。
  - 図-50〜53 砕波帯内の有義波高の算定図 (H1/3)
  - 図-54〜57 砕波帯内の最高波高の算定図 (Hmax = H1/250)
  - 図-61・62 砕波帯内の有義波高・最高波高の略算式と計算値の比較

    cd sample && uv run python compare_goda1975.py
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

import goda_wb.plot as gplot

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "goda1975"

LIST_H0L0 = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
SLOPE_LABELS = ["1/10", "1/20", "1/30", "1/100"]
YLABELS = {"H1_3": "$H_{1/3}/H_0'$", "H1_250": "$H_{max}/H_0'$ ($H_{1/250}$)"}


def draw(column):
    """4 勾配の算定図に厳密計算（実線）と略算式（破線）を重ねる。"""
    chart = gplot.GODA1975_CHARTS[column]
    fig, axes = plt.subplots(2, 2, figsize=(12, 15))
    for k, ax in enumerate(axes.flat):
        gplot.plot_on_goda1975(gplot.CHART_SLOPES[k], LIST_H0L0, column, ax=ax)
        ax.set_title(f"図-{chart['figs'][k]} 海底勾配 {SLOPE_LABELS[k]}  {YLABELS[column]}")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [
        plt.Line2D([], [], color="k", linestyle="-", linewidth=2.5),
        plt.Line2D([], [], color="k", linestyle="--", linewidth=1.8),
    ]
    labels += ["厳密計算", "略算式 (式 67・69)"]
    axes.flat[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(0.12, 0.86), fontsize=9)
    fig.suptitle("合田 (1975) 原図（灰色）と goda_wb の厳密計算（実線）・略算式（破線）")
    fig.tight_layout()
    return fig


def draw_approx():
    """図-61・62 に厳密計算（実線）と略算式（破線）を重ねる。"""
    titles = {
        61: "図-61 有義波高の略算式と計算値 ($H_0'/L_0=0.02$)",
        62: "図-62 最高波高の略算式と計算値 ($H_0'/L_0=0.04$)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for (fig_no, title), ax in zip(titles.items(), axes, strict=True):
        gplot.plot_goda1975_approx_fig(fig_no, ax=ax)
        for line in ax.get_lines():
            line.set_label(line.get_label().replace("computed", "厳密計算").replace("approx", "略算式"))
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("合田 (1975) 原図（灰色：記号が計算値、線が略算式）と goda_wb")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for column in gplot.GODA1975_CHARTS:
        fig = draw(column)
        fig.savefig(OUT_DIR / f"compare_{column}.png", dpi=80)
        plt.close(fig)
    fig = draw_approx()
    fig.savefig(OUT_DIR / "compare_fig61_62.png", dpi=80)
    plt.close(fig)
