#%%
import goda_wb.core as core
from goda_wb.constant import pi2, g
d = 14
T = 12.0
H0 = 12
l0 = g * T**2 / pi2
df = core.cal_surf_goda_dim(tant=0.01, H0=H0, T=T, dim=True)
df
# %%
df.info()
# %%
