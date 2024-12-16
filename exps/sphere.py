import tfdy
import tfdy.plotting as tplt
import torch
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import cm
#%%
d1 = 1.
d2 = 1.
d3 = 0.5

n = 2000
D = torch.diag(torch.tensor([d1, d2, d3], dtype=torch.float))
print(D)
#%%
optim = tfdy.optim_diracs.OptimDiracs(D=D)
x = optim.optimize(n = n, max_it=1000, sigma = 0.1, phi=None, opt_kwargs = {'lr':0.1})

#%%
plt.close('all')
#tplt.set_up_plots()
fig, ax = tplt.fig_ax_3D(figsize=(15,15))

X = tfdy.utils.grid_x_sph(100000)
density = tplt.estimate_density(optim.x(), X)
tplt.vis_sph_particles(
    ax, optim.x(), 
    X, 
    facecolors=cm.coolwarm(density).reshape(X.shape[:2] + (-1,)),
    title = 'Loss Value: ' + str(optim.cur_loss().item())
)

#%%
name = (
        'results/' + 
        reduce(lambda a,b: a+'-'+b, [str(D[i,i].item()) for i in range(3)]) + 
        '.png'
        )
plt.tight_layout()
plt.savefig(name)
