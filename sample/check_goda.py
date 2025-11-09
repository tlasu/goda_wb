#%%
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import goda_wb.core as core
from draw_on_image2 import *

tnat = 0.01


list_h0l0_0 = [ 0.002, 0.003, 0.004, 0.005,
                0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 
                0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

df_dict = {}
for h0l0 in list_h0l0_0:
    dh0 = np.arange(4, 0, -0.05)
    dl0 = [dh0 * h0l0 for dh0 in dh0]
    df = core.cal_surf_goda(tnat, h0l0, dl0)
    df_dict[h0l0] = df

list_h0l0 = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]

decaypoint = []
for h0l0 in list_h0l0_0:
    _ = df_dict[h0l0].query("H1_3 / aks <= 0.98")\
        .sort_values("dh0", ascending=False)\
        .iloc[0]

    x, y = _["dh0"], _["H1_3"]
    decaypoint.append([x, y])


fig = draw_on_image_goda()
ax = fig.axes[1]

for h0l0 in list_h0l0:
    ax.plot(df_dict[h0l0]["dh0"],
            df_dict[h0l0]["H1_3"],
            "-",
            linewidth=3,
            alpha=0.75,
            label=f"$H'_o/L_o={h0l0}$")

ax.plot([_[0] for _ in decaypoint], [_[1] for _ in decaypoint], "-o", linewidth=3)

ax.set_xlim(0,4)
ax.set_ylim(0,3)
ax.set_xticks(np.arange(0, 4.1, 0.5))
ax.minorticks_on()
ax.grid(True)
ax.grid(which="minor", linestyle="--")
ax.legend()
ax.axis("off")
fig.tight_layout()

# %%
