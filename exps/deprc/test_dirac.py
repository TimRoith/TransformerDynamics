import torch
from elliptics import opt_diracs, find_phi_gap, pol2cart, integral_scalar_prod

d1 = .5
d2 = 1
D = 1.* torch.tensor([[d1,0], [0, d2]])

def naive_scalar_prod(d):
    s = 0
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            s+= torch.exp(torch.inner(d[i, :], D@d[j, :]))/(d.shape[0]**2)
    return s
            
d = torch.normal(0, 1., size=(1000, 2))

print(naive_scalar_prod(d) - integral_scalar_prod(D)(d))
            