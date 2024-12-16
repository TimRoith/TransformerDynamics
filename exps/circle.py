import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import tfdy
import tfdy.plotting as tplt
from tfdy.utils import integral_scalar_prod, sph2cart
import torch
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import cm


#%%
d = 2

def proj(x):
    return x/torch.linalg.vector_norm(x, dim=-1,keepdim=True)

@torch.no_grad()
def opt_x(D, beta:float, n:int, tau:float, max_it:int, erg_int:int=50, 
          x = None, d= 2):
    x = proj(torch.randn(n, d)) if x is None else x
    E = integral_scalar_prod(D=D)
    hist = []
    n = x.shape[0]#torch.sum(g, dim=-1)[...,None]
    for i in range(max_it):
        x -= (tau/n) * torch.exp(beta * (x @ (D @ x.T))) @ x @ D.T
        x /= torch.linalg.vector_norm(x, dim=-1,keepdim=True)
        
        if i%erg_int == 0:
            hist.append(E(x))
    return x, hist
#%%
beta=1.
tau = .05
max_it = 1000
n = 100
#%%
Xp = tfdy.utils.init_phi_2D(n)
X = tfdy.utils.pol2cart(Xp)

#%%
plt.close('all')
fig, ax = plt.subplots(1,4)
plt.subplots_adjust(bottom=0.2)


sliders = []
for i in range(1):
    axx = fig.add_axes([0.25, 0.1-0.03*i, 0.65, 0.03])
    sliders.append(Slider(axx, 'd'+str(i), -2, 1, valinit=0.))#
    
sc = []
for i in [0,3]:
    ax[i].plot(X[:,0], X[:,1], c='r')
    ax[i].axis('square')
    sc.append(ax[i].scatter(X[:,0], X[:,1]))
    
l = ax[1].plot(Xp, torch.ones(n)/n, label='Estimate')
l2 = ax[1].plot(Xp, torch.ones(n)/n)
ax[1].set_ylim([-10/n,10/n])

#%%
from scipy.interpolate import interp1d
import numpy as np
def inverse_sample_function(P, pnts, x_min=-100, x_max=100, n=1e5, 
                            **kwargs):
        
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(P(x, **kwargs))
    cumulative -= cumulative.min()
    f = interp1d(cumulative/cumulative.max(), x)
    
    s = np.linspace(0, 1)
    return f(np.random.random(pnts))

class exp_guess:
    def __init__(self, eps=1):
        co = np.array([ 1.15873192e+05,  5.46912620e+04,  1.39741928e+04, -1.13194179e+04,
               -2.52650568e+04, -3.11225153e+04, -3.14659259e+04, -2.82961231e+04,
               -2.31375445e+04, -1.71216247e+04, -1.10584037e+04, -5.49786594e+03,
               -7.82341721e+02,  2.90885862e+03,  5.52155296e+03,  7.09464224e+03,
                7.73313158e+03,  7.58554061e+03,  6.82511979e+03,  5.63437352e+03,
                4.19246872e+03,  2.66517840e+03,  1.19707631e+03, -9.42394030e+01,
               -1.12205675e+03, -1.83270345e+03, -2.20600144e+03, -2.25403621e+03,
               -2.01822069e+03, -1.56461525e+03, -9.77473192e+02, -3.51030513e+02,
                2.20329744e+02,  6.53203744e+02,  8.86914712e+02,  8.94708686e+02,
                6.92520507e+02,  3.42549058e+02, -5.20320100e+01, -3.65402440e+02,
               -4.83702743e+02, -3.54761280e+02, -4.05444191e+01,  2.61881534e+02,
                2.94922329e+02, -2.14727493e+01, -2.74305085e+02,  1.53201899e+02,
               -2.81786781e+01,  3.99737141e+00, -1.28117088e-02])
        f = np.polyval(co, eps)
        #f = torch.exp(torch.pi/2*(eps/(1-eps))**0.5)
        
        
        self.params = f *  np.ones(1)
    
    def __call__(self, theta):
        
        evalp = np.exp(-self.params * np.cos(2 * theta))
        return evalp/np.sum(evalp)

#%%
def sliders_on_changed(val):
    model = exp_guess(eps=sliders[0].val)
    D = torch.diag(torch.tensor([1, 1-sliders[0].val], dtype=torch.float))
    x, hist = opt_x(D, beta, n, tau, max_it, x=None)
    #density = tplt.estimate_density(tfdy.utils.cart2pol(x)[:,None], Xp[:,None])
    density = tplt.estimate_density(x, X, h=None)
    #density = tfdy.utils.pol2cart(torch.tensor(density))
    density *= 1/density.sum()
    sc[0].set_offsets(x)
    
    ax[1].clear()
    ax[1].plot(Xp, 1/n - density)
    
    f = -np.sign(sliders[0].val) * (density - 1/n).max()
    ax[1].plot(Xp, f * torch.cos(2 * Xp))
    ax[1].plot(Xp, -1/n+model(Xp.numpy()))
    ax[1].set_ylim([-10/n,10/n])
    #
    eps = sliders[0].val
    P = torch.tensor(inverse_sample_function(model, n, x_min=-np.pi, x_max = np.pi), dtype=torch.float)
    xp = tfdy.utils.pol2cart(P)
    val = integral_scalar_prod(D=D)(xp)
    
    ax[2].clear()
    ax[2].plot(hist)
    ax[2].plot(np.arange(len(hist)), val * np.ones(shape=(len(hist))))
    
    sc[1].set_offsets(tfdy.utils.pol2cart(P))
    
for s in sliders:
    s.on_changed(sliders_on_changed)

