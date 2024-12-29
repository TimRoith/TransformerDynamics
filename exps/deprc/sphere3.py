import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import tfdy
import tfdy.plotting as tplt
from tfdy.utils import integral_scalar_prod, sph2cart
from tfdy.optim_diracs import usa_flow, proj
import torch
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import cm

    
#%%
beta=1.
tau = .1
max_it = 2000
n = 300

#%%
plt.close('all')
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(131)
axs = fig.add_subplot(132, projection='3d')
axh = fig.add_subplot(133,)
plt.subplots_adjust(bottom=0.2)

sliders = []
for i in range(3):
    axx = fig.add_axes([0.25, 0.1-0.03*i, 0.65, 0.03])
    sliders.append(Slider(axx, 'd'+str(i), -1, 1, valinit=1.))
    
ng = 100
X = tfdy.utils.grid_x_sph(ng**2)
x = proj(torch.randn(n,3))
im = ax.imshow(torch.ones(ng,ng), vmin=0., vmax=1., cmap='coolwarm')
    
    
def sliders_on_changed(val):
    D = torch.diag(torch.tensor([s.val for s in sliders], dtype=torch.float))
    x = torch.zeros((n,3))
    x[:,1] = -1
    x, hist = usa_flow(D, beta, n, tau, max_it, x=None)
    density = tplt.estimate_density(x, X,h=0.1)
    im.set_data(density.reshape(ng,ng).T)
    #sc._offsets3d = (x[:,0], x[:,1], x[:,2])

    axs.clear()
    axs.plot_surface(
        *[X[...,i] for i in range(3)],
        shade=False,
        edgecolor='none',
        facecolors=cm.coolwarm(density.reshape(ng,ng)),
        linewidth=0.1,
        alpha=0.35)
    axs.scatter(x[:,0], x[:,1], x[:,2], alpha=0.4, color='orange',marker='.')
    axs.axis('equal')
    
    axh.clear()
    axh.plot(hist)
    
    fig.canvas.draw_idle()
    plt.show()
    
for s in sliders:
    s.on_changed(sliders_on_changed)