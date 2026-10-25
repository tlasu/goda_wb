"""段階的砕波変形モデル PEGBIS と合田 (1975) のモデルの比較。

合田 (2004) ECOH/YG 技術論文 No.2 の図-6~9（付図-4~7）と同じく、一様勾配斜面上の
Hmax (= H1/250) と H1/3 を比較する。pegbis/compare_slope.png と pegbis/kweon_step.png を出力する。

    cd sample && uv run python compare_pegbis.py
"""

import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np

import goda_wb.core as core
import goda_wb.pegbis as pegbis

dh0 = np.arange(4.0, 0.0, -0.1)
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
for ax, tant in zip(axes.flat, [1 / 10, 1 / 20, 1 / 30, 1 / 100], strict=True):
    for h0l0, color in [(0.02, "C0"), (0.04, "C1")]:
        dl0 = list(dh0 * h0l0)
        ref = core.cal_surf_goda(tant, h0l0, dl0)
        res = pegbis.cal_surf_pegbis_slope(tant, h0l0, dl0, n_freq=10)
        for col, ls in [("H1_250", "-"), ("H1_3", "--")]:
            ax.plot(dh0, ref[col], ls, color=color, label=f"合田(1975) {col} H0/L0={h0l0}")
            ax.plot(dh0, res[col], "o", ms=3, color=color, label=f"PEGBIS {col} H0/L0={h0l0}")
    ax.set_title(f"海底勾配 1/{round(1 / tant)}")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.2)
    ax.grid(True)
for ax in axes[1]:
    ax.set_xlabel("h/H0'")
for ax in axes[:, 0]:
    ax.set_ylabel("H/H0'")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("pegbis/compare_slope.png", dpi=120)

# 権の水平ステップ実験（論文 図-11）
x = np.linspace(-4.0, 4.0, 81)
h = np.interp(x, [-4.0, -3.0, -1.0, 0.0, 4.0], [0.36, 0.36, 0.16, 0.11, 0.11])
df = pegbis.cal_surf_pegbis(x, h, 0.0879, 2.21, n_freq=10)
fig, ax = plt.subplots(figsize=(7, 4))
for col in ["H1_250", "H1_10", "H1_3", "Hrms", "Hmean"]:
    ax.plot(df["x"], df[col], label=col)
ax.plot(x, -h * 0.3, "k", label="水深 ×0.3")
ax.set_xlabel("x (m)")
ax.set_ylabel("波高 (m)")
ax.set_title("水平ステップ上の波高変化 (H0'=0.0879 m, T1/3=2.21 s)")
ax.grid(True)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("pegbis/kweon_step.png", dpi=120)
