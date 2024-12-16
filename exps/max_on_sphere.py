from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod
import matplotlib.pyplot as plt
import torch
import scienceplots
#%%
plt.style.use(['science'])

n = 1
d = 3
kwargs = {
    'max_it': 500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}

mode = 'PD'

if mode == 'Id':
    D = torch.eye(d)
else:
    D = torch.eye(d)
    D[-1,-1] = 10
    D[-2,-2] = 7.5


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
import matplotlib.cm as cm
fig = plt.figure(figsize=(5,5))
axs = fig.add_subplot(111, projection='3d', computed_zorder=False)

import numpy as np
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,reps)))


X = grid_x_sph(50**2)
XX = X.reshape(-1,d)

Z = torch.exp((XX * (XX@D)).sum(axis=-1)).reshape(X.shape[:2])
Z = (Z - Z.min())/(Z.max() - Z.min())

fcolors = torch.zeros(X.shape[:2] + (4,))
lamda = torch.clamp(Z, min=0.1)[:,:,None]
fcolors[..., :3] = (
    (1 - lamda) * torch.tensor([1.,1.,0.9])[None,None,:] + 
    (lamda) * torch.tensor([123, 200, 246])[None,None,:]/255
)

#fcolors[..., :] *= torch.clamp(Z, min=0.1)[:,:,None]
fcolors[..., 3]  = Z#torch.exp(Z-1)
fcolors = torch.clamp(fcolors, max=1, min=0)


sf = axs.plot_surface(
    *[X[...,i] for i in range(3)],
    shade=False,
    edgecolor='k',
    facecolors=fcolors.numpy(),
    linewidth=0.1,
    alpha=.75,
    rstride=1, cstride=1,
    #color='xkcd:off white'
    )
plt.colorbar(sf)
axs.scatter(*[x_all[:,i] for i in [0,1,2]], alpha=1, 
            c='k',
            marker='o',
            s=30
            )
axs.set_xlabel("$e_1$")
axs.set_ylabel("$e_2$")
axs.set_zlabel("$e_3$")

# for i in range(3):
#     axs.plot(*[[-1. * (j==i), 1. * (j==i)] for j in range(3)], 
#              color='k', alpha=0.5)#[-2,2], [0,0],[0,0])

for i in range(3): getattr(axs, 'set_' + chr(120 +i) + 'ticks')(torch.linspace(-1,1,5))
axs.minorticks_off()
# axs.plot([0,0], [-2,2], [0,0])
# axs.plot([0,0],[0,0],[-2,2])
#axs.set_axis_off()
axs.axis('square')
axs.set_xlim([-1.,1.])
axs.set_ylim([-1.,1.])
axs.set_zlim([-1.,1.])





#%%
save = False
if save:
    fname = input('Please specify the filename: \n')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig('./results/' + fname + '.png', dpi=600)


#%%
plt.figure()
plt.plot(hist)