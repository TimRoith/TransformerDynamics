import torch
from scipy.special import gamma
import numpy as np

exp, cos, sin = (torch.exp, torch.cos, torch.sin)

N = 100000
n = 37
phi = torch.linspace(0, torch.pi, N)

I1 = 2 *  (torch.exp(torch.cos(phi)) * 
           torch.sin(phi)**(n-2) * 
           torch.cos(phi)**2
           ).sum()
I2 = (torch.exp(torch.cos(phi)) * torch.sin(phi)**(n)).sum() * (1/(n-1))

print(20*'-')
print(I1/N)
print(I2/N)

#%%
print(20*'-')

I1 = (torch.exp(torch.cos(phi)) * 
      torch.sin(phi)**(n-2) * 
      torch.cos(phi)**2
    ).sum()
I2 = (torch.exp(torch.cos(phi)) * torch.sin(phi)**n).sum()

print(I1/N)
print(I2/N)

#%%
def S(n):
    return 2 * np.pi**(n-2) * (1/gamma(n/2))


I1 = S(n-2) * (exp(cos(phi)) * sin(phi)**(n-2) * cos(phi)**2).sum()/N
I2 = S(n-3) * (cos(phi)**2 * sin(phi)**(n-3)).sum()/N * (exp(cos(phi)) * sin(phi)**n).sum()/N
I3 = S(n-3) * (1/(n-1)) * (sin(phi)**(n-3)).sum()/N   * (exp(cos(phi)) * sin(phi)**n).sum()/N

print(20*'-')
print(I1)
print(I2)
print(I3)

#%%
I1 = ((1-sin(phi)**2) * sin(phi)**(n-3)).sum()/N
I2 = (1/(n-1)) * (sin(phi)**(n-3)).sum()/N

print(20*'-')
print(I1)
print(I2)