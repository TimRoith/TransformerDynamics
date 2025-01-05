#%%
from tfdy.optim_diracs import usa_flow, proj, OptimDiracs, find_decr_step_size
from tfdy.utils import grid_x_sph, integral_scalar_prod, init_phi
from tfdy.coordinates import sph2cart
from tfdy.plotting import PlotConf2D, colored_line
import matplotlib.pyplot as plt
import torch
import tqdm
import numpy as np
#%%

n = 4
d = 2
tau0 = 5
kwargs = {
    'max_it': 10000,
    'tau': 0.05,
    'n':n,
    'beta':1,
    'sigma':0.,
    'sgn':1
}

#%%
reps = 1
num_d = 1
ds = torch.linspace(0.5, 0.5, num_d)
tanh_mean = torch.zeros(num_d)

for i in tqdm.trange(num_d):
    dd = ds[i]
    D = torch.diag(torch.tensor([1, dd]))
    xinit = sph2cart(init_phi(n, d) + torch.pi/n, excl_r=True)

    for r in range(reps):
        #x[..., idx] = torch.abs(x[..., idx])
        x, hist = usa_flow(D, x=xinit.clone(), **kwargs)
        ev = torch.tanh(D[0,0] * x[:,0]**2)/torch.tanh(D[1,1] * x[:,1]**2)
        tanh_mean[i] += ev.mean()
        
        
tanh_mean *= 1/reps

#%%
save_csv = False
if save_csv:
    X = torch.stack([ds, tanh_mean], dim=1)
    np.savetxt('results/tanh_min' + str(kwargs['tau']) + '.csv', X)


#%%
plt.close('all')
plt.figure()
plt.plot(ds, tanh_mean)
plt.plot(ds,ds)

#%%
plt.figure()
plt.plot(hist)


#%%
plt.close('all')
PC = PlotConf2D(figsize=(4, 4))
PC.init_axs(1, lims=(-1.1, 1.1))
PC.plot_colored_circle(D, vmin=0., vmax=10.0, linewidth=10.)
PC.axs[0].axis('off')
PC.axs[0].scatter(x[:, 0], x[:, 1], s=400, color='b', zorder=2)
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
# %%
