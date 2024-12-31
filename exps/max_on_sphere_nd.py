from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod, cc_kmeans
from tfdy.plotting import PlotConf3D
from tfdy.coordinates import cart2sph
import matplotlib.pyplot as plt
import torch
import tqdm

d = 3
n = 100
kwargs = {
    'max_it': 500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}

mode = 'NonId2'

if mode == 'Id':
    torch.manual_seed(142)
    D = -torch.eye(d)
    reps = 6
elif mode == 'NonId':
    D = -torch.eye(d)
    D[-1,-1] = -4
    D[-2,-2] = -3
    reps = 100
elif mode == 'NonId2':
    D = -torch.eye(d)
    D[-1,-1] = 0
    D[-2,-2] = 0
    reps = 10
    
xall = []
csall = []
for r in tqdm.trange(reps):
    x = proj(torch.randn(n ,d))
    x, hist = usa_flow(D, x=x, **kwargs)
    xall.append(x)
    csall.append(cc_kmeans(x)['cs'])
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

fcolors = PC.cmap.reversed()(Z)

az, el = (PC.axs[0].azim/180)*torch.pi, (PC.axs[0].elev/180)*torch.pi,
#fcolors[...,-1] = Z.numpy()#torch.exp(Z-1).numpy()
PC.plot_sphere(facecolors=fcolors, alpha=.7, zorder=-1)
x = torch.stack(xall).reshape(-1, d)
phi = cart2sph(x[..., [2,0,1]], excl_r=True)
# idx  = torch.where(phi[...,0] < (torch.pi - az%(torch.pi)) * 
#                    phi[...,0] > (torch.pi - az%(torch.pi))
#                    )

def lazy_view_idx(x, az, el):
    idx  = torch.where((phi[...,1] <  -az/2) * (phi[...,1] > (-torch.pi-az/2)) *
                       (phi[...,0] < torch.pi - el)
                       +
                       (phi[...,0] < el)
                       )[0]
    idx_c = torch.ones(x.shape[0])
    idx_c[idx] = False
    return idx, torch.where(idx_c)[0]

idx, idx_c = lazy_view_idx(x, az, el)

for ii, a, c in [(idx, 1., 'k'), (idx_c, 0.005, 'r')]:
    PC.axs[0].scatter(*list(x[ii, : ].T),
                c=c,
                marker='o',
                s=40,
                alpha=a,
                zorder=2
                )
# PC.axs[0].scatter(*list(x[idx2[0], idx2[1], : ].T),
#             c='r',
#             marker='o',
#             s=30,
#             linewidths=0.4,
#             alpha=1.,
#             )

for x, cs in zip(xall, csall):
    PC.axs[0].plot(*list(cs.T), alpha=0.5, color='k', zorder=1)
PC.add_colorbar(shrink=0.75,vmin=zmin, vmax=zmax)
PC.save(name='Max' + mode + 'Nd')
#%%
plt.figure()
plt.plot(hist)