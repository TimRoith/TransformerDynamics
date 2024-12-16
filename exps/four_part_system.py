from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod, init_phi
from tfdy.coordinates import sph2cart
import matplotlib.pyplot as plt
import torch
#%%

n = 8
d = 2
kwargs = {
    'max_it': 20500,
    'tau': 0.0005,
    'n':n,
    'beta':1,
    'sigma':0.,
    'sgn':1
}


#%%
D = torch.diag(torch.tensor([1., 3.]))
x = sph2cart(init_phi(n, d)[...,None] + torch.pi/n, excl_r=True)
#x[..., idx] = torch.abs(x[..., idx])
x, hist = usa_flow(D, x=x, **kwargs)


print(torch.tanh(D[0,0] * x[:,0]**2)/torch.tanh(D[1,1] * x[:,1]**2))
print(D[1,1]/D[0,0])
#%%
plt.plot(hist)


#%%
from matplotlib.patches import Circle
plt.close('all')
fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1])
#plt.scatter(s[:, 0], s[:, 1])
ax.add_patch(Circle((0,0), 1, fill =False))

ax.axis('equal');
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
