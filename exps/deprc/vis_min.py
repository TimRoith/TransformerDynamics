import torch
from elliptics import opt_diracs, find_phi_gap, pol2cart, cart2pol, get_uniform_ellipse,integral_scalar_prod
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

d1 = 0.5
d2 = 1
a = 1/(d1**0.5)
D = 1.* torch.tensor([[d1,0], [0, d2]])
n = 2000
phi = torch.linspace(-torch.pi, torch.pi, n+1)[:-1]# + torch.normal(0, 0.01, size=(n,))
phi_proj = torch.linspace(-torch.pi, torch.pi, n//2+3)[:-1]

xproj = torch.cat([s * pol2cart(torch.atan(torch.tan(phi) * (d1**(3/2)))) for s in [1,-1]])
xproj = xproj[torch.where(xproj.abs()[:,1]>0.01)]
phi = cart2pol(xproj)

a_opt = 1.
x, hist = opt_diracs(D, n = n, lr = 0.01, max_it=1000, a=a_opt, phi=phi, sigma=0.01)

#%%
plt.close('all')
fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1], marker='.')
#ax.scatter(xproj[:, 0], xproj[:, 1], marker='.', alpha=0.5)

ellipse = Ellipse((0,0), 2 * a_opt, 2/(d2**0.5), angle=0, fill=False)
ax.add_patch(ellipse)

ax.axis('equal')

    