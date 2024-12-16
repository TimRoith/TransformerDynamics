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
    'sigma':0.
}


#%%
eps = -3.5
reps = 1
idx = -1

D = torch.eye(d)
D[-1,-1] = -2
x = proj(torch.randn(n ,d))
#x[..., idx] = torch.abs(x[..., idx])
x, hist = usa_flow(D, x=x, **kwargs)

#%%
plt.close('all')
fig = plt.figure(figsize=(5,5))
axs = fig.add_subplot(111, projection='3d', computed_zorder=False)

X = grid_x_sph(50**2)
axs.plot_surface(
    *[X[...,i] for i in range(3)],
    shade=False,
    edgecolor='k',
    linewidth=0.1,
    alpha=.75,
    rstride=1, cstride=1,
    color='xkcd:ice blue')
axs.scatter(*[x[:,i] for i in [0,1,2]], alpha=1, 
            color='xkcd:reddish',
            marker='o',
            s=30
            )
axs.set_xlabel("$e_1$")
axs.set_ylabel("$e_2$")
axs.set_zlabel("$e_3$")

axs.set_xlim([-1,1])
axs.set_ylim([-1,1])
axs.set_zlim([-1,1])
axs.axis('square')
#%%
E = integral_scalar_prod(D)
print('Final energy: ' + str(E(-x).item()))

yy = torch.zeros_like(x)
yy[..., :yy.shape[-1]//2] = 1
yy[..., yy.shape[-1]//2:] = 1

print('Final energy: ' + str(E(yy).item()))


#%%

D = -torch.eye(2)
D[-1,-1] = -1
E = integral_scalar_prod(D)
def sym(x, e):
    return x - 2 * (x*e).sum(dim=-1, keepdim=True) * e

z = torch.zeros((1,2))
z[:,0] = 1


for i in range(10000):
    xx = torch.randn(size=(2,2))
    #xx = torch.ten
    
    xx = xx/torch.linalg.norm(xx, axis=-1, keepdims=True)
    sx = sym(xx,z)
    print(E(xx))
    print(E(sx))
    #s = 0.5*(xx + sym(xx,z))
    #s = s/torch.linalg.norm(s, axis=-1, keepdims=True)
    s = torch.cat([xx, sx])
    print(E(s))
    if torch.min(torch.linalg.norm(xx-sx), torch.linalg.norm(xx[[1,0], :]-sx)) > 1e-2:
        assert E(s) >= E(xx)

#%%
from matplotlib.patches import Circle
plt.close('all')
fig, ax = plt.subplots()
ax.scatter(xx[:, 0], xx[:, 1])
ax.scatter(sx[:, 0], sx[:, 1])
#plt.scatter(s[:, 0], s[:, 1])
ax.add_patch(Circle((0,0), 1, fill =False))

ax.axis('equal');
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
#%%
plt.figure()
plt.plot(hist)