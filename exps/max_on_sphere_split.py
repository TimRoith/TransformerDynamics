from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
from tfdy.utils import grid_x_sph, integral_scalar_prod
from tfdy.plotting import PlotConf3D
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.cluster.vq import kmeans
import tqdm

d = 2
n = 100
kwargs = {
    'max_it': 1500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.
}

#%%
def get_biggest_eigvector(D):
    l,v = torch.linalg.eigh(D)
    return v[:, -1]

def count_close(x, v, tol=1e-1):
    n = x.shape[0]
    nums = [(torch.linalg.norm(v - sgn * x, dim=-1) < tol).sum() for sgn in [-1,1]]
    
    res = {'weights': torch.tensor(nums)}
    if sum(nums) != n:
        res['cc'] = 2
    elif (nums[0] * nums[1]) == 0:
        res['cc'] = 0
    else:
        res['cc'] = 1
    return res

def cc_kmeans(x, tol=0.2):
    cs, v = kmeans(x, 3)
    if cs.shape[0] > 1:
        s = (np.linalg.norm(cs[None, ...] - cs[:,None, :],axis=-1) > 0.1).sum()//(cs.shape[0] - 1)
    else: s = 1
    return {'cc': max(s - 1, 0)}
    
    
#%%
reps = 100
num_ds = 20

CCs = []
Es  = torch.zeros(num_ds)
TEs = torch.zeros(num_ds)
ProdMin = torch.zeros((num_ds, reps))


ds = torch.linspace(0.0, 1.5, num_ds)
for i in tqdm.trange(num_ds):
    dd = ds[i]
    D = torch.diag(torch.tensor([dd, 1.]))
    E = integral_scalar_prod(D)
    v = get_biggest_eigvector(D)
    evals, w = torch.zeros(3), torch.zeros(2)
    
    for r in range(reps): 
        x = proj(torch.randn(n ,d))
        x, hist = usa_flow(D, x=x, **kwargs)
        res = cc_kmeans(x)
        evals[res['cc']] += 1
        Es[i] += E(x)
        
        if dd == 1 and (res['cc'] != 0):
            raise ValueError
        elif res['cc'] > 1:
            raise ValueError
        
        ProdMin[i, r] = (x[None, ...] * x[:, None, :]).sum(-1).min().abs()
    # test out single Dirac and normalize
    x[:] = v
    TEs[i] = E(x)
    Es[i] *= 1/reps
        
    CCs.append(evals)
#%%
CC = np.concatenate([ds.numpy()[:,None], np.array(CCs)], axis=-1)

for name, z in [('CC', CC), ('Es', Es), ('TEs', TEs), ('Pmin', ProdMin)]:
    np.savetxt('results/split_Eval/' + name + '.txt', z)


# for i in range(E.shape[-1]):
#     np.savetxt('results/split_Eval/'+str(i) +'.txt', E[:, i])