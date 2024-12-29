from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod, init_phi_2D
from tfdy.coordinates import sph2cart
from tfdy.plotting import PlotConf2D, colored_line
import matplotlib.pyplot as plt
import torch
import numpy as np

d = 2
n = 100
kwargs = {
    'max_it': 1400,
    'tau': 0.075,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}

D = torch.diag(torch.tensor([1., 1.25]))

x = proj(torch.randn(n ,d))
x, hist = usa_flow(D, x=x, **kwargs)

#%%
plt.close('all')
PC = PlotConf2D()
PC.init_axs(1, lims=(-1.1, 1.1))
PC.plot_colored_circle(D, vmin=1., vmax=1.4, linewidth=10.)
PC.axs[0].axis('off')
PC.axs[0].scatter(x[:, 0], x[:, 1], s=400, color='b', zorder=2)

save = False
if save: PC.save()
#%%
#plt.figure()
#plt.plot(hist)