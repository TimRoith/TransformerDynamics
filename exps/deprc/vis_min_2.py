import torch
from elliptics import opt_diracs, find_phi_gap, pol2cart, cart2pol, get_uniform_ellipse,integral_scalar_prod
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

d1 = 0.5
d2 = 1
a = 1/(d1**0.5)
D = 1.* torch.tensor([[d1,0], [0, d2]])
n = 200
phi = torch.linspace(-torch.pi, torch.pi, n+1)[:-1]# + torch.normal(0, 0.01, size=(n,))
phi_proj = torch.linspace(-torch.pi, torch.pi, n//2+3)[:-1]

xproj = torch.cat([s * pol2cart(torch.atan(torch.tan(phi) * (d1**(3/2)))) for s in [1,-1]])
xproj = xproj[torch.where(xproj.abs()[:,1]>0.01)]
phi = cart2pol(xproj)

#%%
a_opt = 1.
x, hist = opt_diracs(D, n = n, lr = 0.01, max_it=100, a=a_opt, phi=phi, sigma=0.01)
#%%
L = integral_scalar_prod(D)
#p = torch.tensor([0.8165, 0.5774])
p = torch.tensor([((1-d1)/(1+d1))**0.5, 0.5774])
p[1] = (1 - p[0]**2)**0.5

phi = cart2pol(p)
xp = pol2cart(torch.linspace(-phi, phi, n//2))
xp = torch.cat([sgn * xp for sgn in [1,-1]])

xunif = pol2cart(torch.linspace(-torch.pi, torch.pi, n+1)[:-1])

plt.close('all')
fig, ax = plt.subplots()
ax.scatter(xp[:, 0], xp[:, 1], marker='.')
ax.scatter(*p, color='r',)


ellipse = Ellipse((0,0), 2 * a_opt, 2/(d2**0.5), angle=0, fill=False)
ax.add_patch(ellipse)

ax.axis('equal')

print('unif: ' + str(L(xunif).item()))
print('xp: ' + str(L(xp).item()))


#%%
plt.close('all')
fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1], marker='.')
ax.scatter(*p, color='r',)
#ax.scatter(xproj[:, 0], xproj[:, 1], marker='.', alpha=0.5)

ellipse = Ellipse((0,0), 2 * a_opt, 2/(d2**0.5), angle=0, fill=False)
ax.add_patch(ellipse)

ax.axis('equal')

    