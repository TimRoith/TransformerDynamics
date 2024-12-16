from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod
import matplotlib.pyplot as plt
import torch
import scienceplots
#%%
plt.style.use(['science'])

n = 100
d = 3
kwargs = {
    'max_it': 500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}

mode = 'Id'

if mode == 'Id':
    eps = 0
else:
    eps = -1


#%%
reps = 10
x_all = torch.zeros(reps, d)
idx = -1
for k in range(reps):
    D = torch.eye(d)
    D[-1,-1] = 1 - eps
    x = proj(torch.randn(n ,d))
    #x[..., idx] = torch.abs(x[..., idx])
    x, hist = usa_flow(D, x=x, **kwargs)
    x_all[k, :] = torch.mean(x, dim=0)

#%%
plt.close('all')
import matplotlib.cm as cm
fig = plt.figure(figsize=(5,5))
axs = fig.add_subplot(111, projection='3d', computed_zorder=False)

import numpy as np
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,reps)))


X = grid_x_sph(50**2)
axs.plot_surface(
    *[X[...,i] for i in range(3)],
    shade=False,
    edgecolor='k',
    linewidth=0.1,
    alpha=.75,
    rstride=1, cstride=1,
    color='xkcd:ice blue')
axs.scatter(*[x_all[:,i] for i in [0,1,2]], alpha=1, 
            c=cm.Accent([range(reps)])[0,...],
            marker='o',
            s=30
            )
axs.set_xlabel("$e_1$")
axs.set_ylabel("$e_2$")
axs.set_zlabel("$e_3$")

axs.set_xlim([-1,1])
axs.set_ylim([-1,1])
axs.set_zlim([-1,1])

axs.set_axis_off()
axs.axis('square')
#%%
save = False
if save:
    fname = input('Please specify the filename: \n')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig('./results/' + fname + '.png', dpi=600)


#%%
plt.figure()
plt.plot(hist)