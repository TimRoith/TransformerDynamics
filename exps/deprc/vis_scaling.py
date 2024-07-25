import torch
import torch.nn as nn
from elliptics import opt_diracs, find_phi_gap, pol2cart, cart2pol, get_uniform_ellipse,integral_scalar_prod
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def scale_phi(s, n =10):
    return torch.cat([s/2*torch.linspace(-torch.pi, torch.pi, n//2+1)[:-1] + a for a in [0,torch.pi]]) + torch.pi/2
    
#%%

def get_s(d1 = 1., d2 = 1, a_opt=1, n=300):
    D = 1.* torch.tensor([[d1,0], [0, d2]])
    s = torch.ones((1,))
    s = nn.Parameter(s)
    opt = torch.optim.Adam([s], lr=0.1 * d1**0.5)
    loss_fct = integral_scalar_prod(D)
    for i in range(500):
        opt.zero_grad()
        loss = loss_fct(pol2cart(scale_phi(s, n=n), a=a_opt))
        loss.backward()
        opt.step()
        #sched.step(loss)
    return s.abs().item(), loss.item()
#%%
cs = []
ds = torch.linspace(1,15, 20)
for d1 in ds:
    s, l = get_s(d1=d1)
    cs.append(s)
    print(10*'-')
    print(s)
    print(l)

#%%
plt.close('all')
plt.plot(ds, cs)
plt.plot(ds, torch.exp(-(ds-1)**0.5))

