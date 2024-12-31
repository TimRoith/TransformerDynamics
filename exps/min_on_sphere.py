#%%
from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod
from tfdy.plotting import PlotConf3D, lazy_view_idx
import matplotlib.pyplot as plt
%matplotlib ipympl
import torch

#%%
d = 3
torch.manual_seed(42) # fix seed to get same initial states

mode = 'PPd'

if mode == 'Id':
    n = 400
    D = torch.eye(d)
    cut_off_value_show = -torch.inf
    tau = 0.1
elif mode == 'PPd':
    D = torch.eye(d)
    D[1, 1] = 0.25
    n = 400
    tau = 0.5
elif mode == 'Sd':
    D = torch.eye(d)
    D[1, 1] = 0
    n = 400
    tau = 0.75
elif mode == 'VSd':
    D = torch.eye(d)
    D[1, 1] = 0
    D[0, 0] = 0
    n = 400
    tau = 0.75
else:
    raise ValueError('Unknown mode')

#%%
kwargs = {
    'max_it': 20000,
    'tau': tau,
    'n':n,
    'beta':1,
    'sgn': 1,
    'sigma':0.
}
#%%

x = proj(torch.randn(n ,d))
x, hist = usa_flow(D, x=x, **kwargs)


#%%
plt.close('all')
PC = PlotConf3D()
PC.init_axs(1, labelpad=-10, show_ticks=False, axlabelfontsize=25)

X = grid_x_sph(50**2)
XX = X.reshape(-1,d)

Z = ((XX * (XX@D)).sum(axis=-1)).reshape(X.shape[:2])
zmin, zmax = Z.min(), Z.max()
if (zmax - zmin).abs() > 1e-3:
    Z = (Z - zmin)/(zmax - zmin)
else:
    zmin = 0
    zmax*=2
    Z/=zmax

fcolors = PC.cmap(Z)
#fcolors[...,-1] = Z.numpy()#torch.exp(Z-1).numpy()
PC.plot_sphere(facecolors=fcolors, alpha=.9, zorder=-1)

idx1, idx2 = lazy_view_idx(x, PC.axs[0])

for a, idx in ((0.9, idx1), (0.05, idx2)):
    PC.axs[0].scatter(*[x[idx,i] for i in [0,1,2]],
                c='k',
                marker='o',
                s=30,
                linewidths=0.4,
                alpha=a,
                )
#PC.axs[0].view_init(elev=0, azim=-55, roll=0)
cbar = PC.add_colorbar(shrink=0.75,vmin=zmin, vmax=zmax, 
                orientation='horizontal', pad=-0.,)
cbar.ax.tick_params(labelsize=18) 
PC.save(name='Min'+mode)
#plt.show()
# %%
