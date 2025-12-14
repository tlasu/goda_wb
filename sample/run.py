# %%
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib
import goda_wb.core as core
from draw_on_image2 import *
from matplotlib.ticker import MultipleLocator

tnat = 0.01
target_h0l0 = [0.035]

site_h0l0 = 0.035
site_dh0 = 1.1
site_dl0 = site_dh0 * site_h0l0

gragh_config = {
    "xlim": (0, 20),
    "ylim": (0, 3),
    "yticks": np.arange(0, 3.1, 0.5),
}

list_h0l0_0 = list(
    set(
        [
            0.002,
            0.003,
            0.004,
            0.005,
            0.01,
            0.02,
            0.025,
            0.03,
            0.035,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.1,
        ]
        + target_h0l0
    )
)
list_h0l0_0.sort()

df_dict = {}
for h0l0 in list_h0l0_0:
    dh0 = np.arange(100, 0, -0.05)
    dl0 = [dh0 * h0l0 for dh0 in dh0]
    df = core.cal_surf_goda(tnat, h0l0, dl0)
    df_dict[h0l0] = df

list_h0l0 = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]


# %%
xvar = "dh0"
yvar = "H1_3"

decaypoint = []
for h0l0 in list_h0l0_0:
    _ = (
        df_dict[h0l0]
        .query("H1_3 / aks <= 0.98")
        .sort_values("dh0", ascending=False)
        .iloc[0]
    )

    x, y = _[xvar], _[yvar]
    decaypoint.append([x, y])


fig = plt.figure(figsize=(6, 8))
ax = plt.axes()
for h0l0 in list_h0l0:
    ax.plot(
        df_dict[h0l0][xvar],
        df_dict[h0l0][yvar],
        "-",
        linewidth=1,
        alpha=0.5,
        label=f"$H'_o/L_o={h0l0}$",
    )

# ax.plot([_[0] for _ in decaypoint],
#         [_[1] for _ in decaypoint], "--",
#         linewidth=3,
#         alpha=0.5,
#         color="black",
#         label="98% decay line")

# for h0l0 in target_h0l0:
#     ax.plot(df_dict[h0l0][xvar],
#             df_dict[h0l0][yvar],
#             "-",
#             linewidth=3,
#             color="red",
#             label=f"$H'_o/L_o={h0l0}$")

# site = core.cal_surf_goda_point(tnat, site_h0l0, site_dl0)
# ax.scatter(site[xvar], site[yvar],
#            color="red",
#            s=100,
#            marker="o",
#            label="Site")

ax.plot(list_dl0 / 0.02, aks_dict["0.02"], label="Goda's model")
ax.plot(list_dl0 / 0.08, aks_dict["0.08"], label="Goda's model")


if xvar == "dh0":
    ax.set_xlim(gragh_config["xlim"])
elif xvar == "dl0":
    ax.set_xlim(0, 1)

ax.set_ylim(0, 3)
# ax.set_xticks(np.arange(0, gragh_config["xlim"][1] + 0.1, 1))
# ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.grid(True)
ax.grid(which="minor", linestyle="--")
ax.legend()
ax.set_title(f"Goda's model $\\tan\\theta={tnat}$")
ax.set_xlabel("水深波高比 $h/H_o$")
ax.set_ylabel("波高比 $H_{1/3}'/H_o$")
ax.text(
    0.002,
    0.002,
    f"Site: $h/H_o={site['dh0'].values[0]:.3f}$\n$H_{{1/3}}'/H_o={site['H1_3'].values[0]:.3f}$",
)


fig.tight_layout()
fig.savefig("goda.png")

# %%
# list_dl0 = np.arange(0.001, 1, 0.001)
# list_h0l0 = [1.e-9, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

aks_dict = {}
for i, h0l0 in enumerate(list_h0l0_0, start=1):
    aks_list = []
    for dl0 in list_dl0:
        aks = goda_wb.core.shoal(dl0, h0l0)
        aks_list.append(aks)
    aks_dict[f"{h0l0}"] = aks_list

# %%
# %%
for h0l0 in list_h0l0_0:
    plt.plot(
        list_dl0 / h0l0, aks_dict[f"{h0l0}"], label=f"Goda's model $H'_o/L_o={h0l0}$"
    )
plt.xlim(0, 1)
plt.xlabel("水深沖波波長比 $h/L_o$")
plt.ylabel("浅水係数 $K_s$")
plt.legend()
plt.show()
# %%
0.00833
# %%
1 / 120
# %%
