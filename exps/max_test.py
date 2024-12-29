from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod, init_phi_2D
from tfdy.coordinates import sph2cart
from tfdy.plotting import PlotConf2D, colored_line
import matplotlib.pyplot as plt
import torch
import numpy as np

d = 2
n = 20
kwargs = {
    'max_it': 1400,
    'tau': 0.075,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}

D = torch.eye(d)

x  = proj(torch.randn(n ,d))
xs = -1 * x.clone()
x0 = torch.cat([x,xs])

#%%
x, hist = usa_flow(D, x=x0, **kwargs)
#%%
plt.plot(hist)

#%%
plt.close('all')
PC = PlotConf2D()
PC.init_axs(1, lims=(-1.1, 1.1))
PC.plot_colored_circle(D, vmin=1., vmax=1.4, linewidth=5.)
PC.axs[0].axis('off')
PC.axs[0].scatter(x[:, 0], x[:, 1], s=200, color='b', zorder=2)
PC.axs[0].scatter(x0[:, 0], x0[:, 1], s=200, color='r', zorder=2)