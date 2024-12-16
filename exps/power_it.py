import torch
from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
import matplotlib.pyplot as plt
import numpy as np
#%%

n = 5
d = 100
kwargs = {
    'max_it': 500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.,
    'erg_int':1
}

#%%

#%%
class track_dist:
    def __init__(self, v_max):
        self.dists = []
        self.v_max = v_max
        
    def __call__(self, x):
        self.dists.append(torch.linalg.norm((x - self.v_max), dim=-1).min())

def power_it(D, max_it=100, sint=50):
    x_pow = torch.ones(D.shape[-1])
    pow_dist = []
    for i in range(max_it):
        xx = D@x_pow
        x_pow = xx/torch.linalg.norm(xx)
        if i%sint == 0:
            pow_dist.append(torch.linalg.norm(x_pow - v_max))
    return x_pow, pow_dist

#%%
mode = 'transformer'
reps = 100
dists = 0

for rep in range(reps):
    D = torch.randn(size=(d,d)) + 1.5 * torch.eye(d)
    D = 0.5* (D+ D.T)
    l, v = torch.linalg.eigh(D)
    idx = torch.argmax(l.abs())
    v_max = v[:, idx]
    
    x = proj(torch.randn(n ,d))
    if mode == 'transformer':
        td = track_dist(v_max)
        x, hist = usa_flow(D, x=x, track_fct=td, **kwargs)
        dists_loc = torch.tensor(td.dists)
    else:
        x, hist = power_it(D, max_it = kwargs['max_it'], sint=1)
        dists_loc = torch.tensor(hist)
        
    dists = (rep * dists + dists_loc)/(rep + 1)
    

#%%
np.savetxt('results/' + mode + '_power_it.txt', dists.numpy())
    
    

#%%
plt.close('all')
plt.figure()
plt.loglog(dists)