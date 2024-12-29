from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod, init_phi_2D
from tfdy.coordinates import sph2cart
from tfdy.plotting import PlotConf2D, colored_line
import matplotlib.pyplot as plt
import torch
import numpy as np

d = 2
n = 2
kwargs = {
    'max_it': 1400,
    'tau': 0.075,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}



#%%
sc, tcc, tc = [torch.zeros((n,d)) for _ in range(3)]
sc[:, 1] = 1

tc[0, 0] =  1
tc[1, 0] = -1

tcc[0, 1] =  1
tcc[1, 1] = -1

num_d = 100
Es = torch.zeros(3, num_d)

ds = torch.linspace(-1, 1, num_d)
for i,dd in enumerate(ds):
    D = torch.diag(torch.tensor([-1., dd]))
    E = integral_scalar_prod(D)
    
    for j,z in enumerate([sc, tc, tcc]):
        Es[j, i] = E(z)
    
#%%
plt.close('all')
plt.plot(ds, Es[1, :], label='Two Cluster')
plt.plot(ds, Es[0, :], label='Single Cluster')
plt.plot(ds, Es[2, :], label='Other two Cluster')
plt.legend()

#%%
plt.close('all')
for i, (z, c) in enumerate([(sc, 'b'), (tc, (0.0, 0.42, 0.4)), (tcc, 'r')]):
    PC = PlotConf2D()
    PC.init_axs(1, lims=(-1.1, 1.1))
    PC.plot_colored_circle(D, vmin=-1., vmax=2., linewidth=10.)
    PC.axs[0].axis('off')
    PC.axs[0].scatter(z[:, 0], z[:, 1], s=400, color=c, zorder=2)
    PC.save(name='maxid' + str(i))

#%%
np.savetxt('results/maxid.csv',torch.cat([ds[None,:],Es]).numpy().T, delimiter=" ")

