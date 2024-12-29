import torch
from numpy import sin, cos, exp, pi
from tfdy.utils import integral_scalar_prod

def sym(x, e):
    return x - 2 * (x*e).sum(dim=-1, keepdim=True) * e

def Emutilde(phi, l=-1):
    c = cos(phi)**2
    out = (
    exp( (( l - 1) * c + 1)) +
    exp(-(( l - 1) * c + 1)) +
    exp( ((-l - 1) * c + 1)) +
    exp(-((-l - 1) * c + 1))
    )
    return out/4

def Emu(phi, l):
    c = cos(phi)**2
    out = (
        exp( (( l - 1) * c + 1)) +
        exp(-(( l - 1) * c + 1))
    )
    return out/2
        
#%%
z = torch.zeros((1,2))
z[:,1] = 1


phi = 0.2
xx = torch.tensor([[cos(phi), sin(phi)], [cos(pi+phi), sin(pi+phi)]], dtype=torch.float)
sx = sym(xx,z)

s = torch.cat([xx, sx])

#%%
l = -2.
D = torch.eye(2)
D[0,0] = l
E = integral_scalar_prod(D)

print(E(xx))
print(E(s))
    