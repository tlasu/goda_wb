#%%
import goda_wb
import numpy as np
import matplotlib.pyplot as plt
from draw_on_image import draw_on_image

list_dl0 = np.arange(0.001, 1, 0.001)
list_h0l0 = [1.e-9, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

aks_dict={}
for i, h0l0 in enumerate(list_h0l0, start=1):
    aks_list = []
    for dl0 in list_dl0:
        aks = goda_wb.core.shoal(dl0, h0l0)
        aks_list.append(aks)
    aks_dict[f"{h0l0}"] = aks_list

fig = draw_on_image()
fig.axes[1]
plt.title("首藤らの浅水係数計算結果との比較")
plt.xlabel("水深沖波波長比 $h/L_o$")
plt.ylabel("浅水係数 Ks = $H'/H_o$")
for key in aks_dict.keys():
    if float(key) < 0.00001:
        plt.plot(list_dl0, aks_dict[key], label="微小振幅波", c="k",
         alpha=0.5,
         linestyle="--",
         linewidth=4)
    else:
        plt.plot(list_dl0, aks_dict[key],
         label=f"$H'_o/L_o={key}$",
         alpha=0.5,
         linewidth=4)
plt.xlim(0.001, 1)
plt.ylim(0,4)
plt.xscale("log")
plt.legend(title="計算結果",loc="lower left")
plt.axis("off")
# %%
