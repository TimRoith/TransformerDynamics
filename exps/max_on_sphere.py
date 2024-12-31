from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod
from tfdy.plotting import PlotConf3D
import matplotlib.pyplot as plt
%matplotlib ipympl
import torch

d = 3

mode = 'NonId'

if mode == 'Id':
    n = 1
    D = torch.eye(d)
    cut_off_value_show = -torch.inf
else:
    D = torch.eye(d)
    D[-1,-1] = 4
    D[-2,-2] = 3
    cut_off_value_show = -0.8
    n = 1



kwargs = {
    'max_it': 500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}
#%%
reps = 30
x_all = torch.zeros(reps, d)
idx = -1
for k in range(reps):
    x = proj(torch.randn(n ,d))
    #x[..., idx] = torch.abs(x[..., idx])
    x, hist = usa_flow(D, x=x, **kwargs)
    x_all[k, :] = torch.mean(x, dim=0)


#%%
plt.close('all')
PC = PlotConf3D()
PC.init_axs(1)


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

idx1 = torch.where(x_all[:, -1] >= cut_off_value_show)[0]
idx2 = torch.where(x_all[:, -1] < cut_off_value_show)[0]

for a, idx in ((0.9, idx1), (0.01, idx2)):
    PC.axs[0].scatter(*[x_all[idx,i] for i in [0,1,2]],
                c='k',
                marker='o',
                s=30,
                linewidths=0.4,
                alpha=a,
                )
#PC.axs[0].view_init(elev=0, azim=-55, roll=0)
PC.add_colorbar(shrink=0.75,vmin=zmin, vmax=zmax)
#%%
PC.save(name=mode)