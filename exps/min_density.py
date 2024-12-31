#%%
from tfdy.optim_diracs import optimizer_density_2D
from tfdy.utils import ( grid_x_sph,
    init_phi_2D, weighted_integral_scalar_prod,
    perturb_density)
from tfdy.coordinates import sph2cart
from tfdy.plotting import PlotConf3D, colored_line, setup_style
import matplotlib.pyplot as plt
%matplotlib ipympl
import torch
import os
import numpy as np
from tfdy.plotting import get_surf_colormap, fill_between_3d
from tfdy.utils import exp_pdf_guess

#%%
ds = torch.linspace(0.9, 1.3, 9)
N = 1000
rerun_all = False
if rerun_all:
    for i,d2 in enumerate(ds):
        D = torch.diag(torch.tensor([1., d2]))
        w, hist = optimizer_density_2D(D, N=N, max_it = 1500)

        print(20 * '-')
        print('Finished with d2 = ' + str(d2))
        print('Final loss: ' + str(hist[-1]))

        np.savetxt('results/densities/MinDensity_d2_' + str(i) + '.txt', w.detach().numpy())
    np.savetxt('results/densities/ds.txt', ds.numpy())

#%% Load results
Ws = []
for file in os.listdir('results/densities/'):
    filename = os.fsdecode(file)
    if filename.endswith(".txt") and not filename == 'ds.txt':
        Ws.append(np.loadtxt('results/densities/' + filename))
Ws = np.array(Ws)
Ws = np.roll(Ws, Ws.shape[1]//4, axis=-1)
Ws  = torch.tensor(Ws)
M   = torch.diag(torch.tensor([0., 1.]))
pd  = perturb_density(M, N)
#%% Plot results
setup_style()
plt.close('all')
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d', computed_zorder=True)
ax.view_init(elev=40, azim=-40)

x   = torch.linspace(-torch.pi/2, 2 * np.pi -torch.pi/2, N)
X,Y = np.meshgrid(x, ds - 1)

# labels 
ax.set_xlim([X.min(), X.max()])
ax.set_xticks([0, np.pi])
ax.set_xticklabels([r'0', r'$\pi$'])
ax.set_ylim([ds[0] - 1, ds[-1] - 1])
ax.set_zlim([0, 1.1 * Ws.max()])
ax.set_zticks([])
ax.minorticks_off()
ax.set_xlabel(r'$\varphi$', labelpad=-15, rotation = 0, fontsize=15)
ax.set_ylabel(r'$\varepsilon$', labelpad=-15, rotation = 0, fontsize=15)
ax.set_zlabel(r'Density $m$', labelpad=-15, rotation = 0, fontsize=12)


for i in range(Ws.shape[0]):
    c = get_surf_colormap()(.3 * i/Ws.shape[0])
    ax.plot(x, ds[i] - 1, Ws[i, ...], c=c, linewidth=1.5, zorder=0)#
    plot_other_lines = False
    if plot_other_lines:
        ax.plot(x, torch.ones_like(Ws[i, ...]) * ds[i] - 1, 
                pd.eval_sph_coord(x, eps = ds[i] - 1),
                color='b')
        ax.plot(x, torch.ones_like(Ws[i, ...]) * ds[i] - 1,
                exp_pdf_guess(x, eps=ds[i] - 1),
                color='r')
    fill_between_3d(ax, 
                    x, torch.ones_like(Ws[i, ...]) * ds[i] - 1, Ws[i, ...],
                    x, torch.ones_like(Ws[i, ...]) * ds[i] - 1, torch.zeros_like(Ws[i, ...]),
                    c=c,
                    alpha=.5,
                    zorder=(Ws.shape[0] - i)*10,
                    mode=1
                    )

ax.set_box_aspect([1.0, 1.75, 1])
#ax.view_init(elev=30, azim=-90)

#%%
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.savefig('results/densities/MinDensity.png', dpi=600)

#%% Eval diffs
PWs = torch.zeros_like(Ws)
diffs = torch.zeros(Ws.shape[0], 4)
r = torch.normal(0, 1, (N,100))
r = r - r.sum(0, keepdim=True) / N

for i, d in enumerate(ds):
    PWs[i, :] = pd.eval_sph_coord(x, eps = d - 1)
    diffs[i, 0] = torch.linalg.norm(PWs[i, :] - Ws[i, :], ord=2)
    diffs[i, 1] = torch.linalg.norm(exp_pdf_guess(x, eps=d - 1) - Ws[i, :], ord=2)
    #diffs[i, 2] = torch.linalg.norm(1/N + (d - 1) * r - Ws[i, :, None], ord=2, dim=0).mean()
    if not np.isclose(d, 1):
        diffs[i, 3] = torch.linalg.norm(pd.nu_star_sph(x) - (Ws[i, :] - 1/N)/(d-1+1e-14), ord=2)
    else:
        diffs[i, 3] = 0.
np.savetxt('results/densities/diffs.csv', torch.cat([ds[:,None]-1, diffs], dim=-1).numpy())
#%%
plt.figure()
for i in range(diffs.shape[-1]): plt.plot(ds-1, diffs[:, i])
# %%
