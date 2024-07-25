import torch
from elliptics import opt_diracs, find_phi_gap, pol2cart
import matplotlib.pyplot as plt

mode = 'shear' #'reflexion'
a = 5

if mode == 'reflexion':
    D = 1.*torch.tensor([[1,0], [0,-1]])
elif mode == 'shear':
    D = 1.*torch.tensor([[1,0], [0, a]])
else:
    D = torch.eye(2)
    
n = 100
phis = []

for a in torch.linspace(1,10, 20):
    D = 1.*torch.tensor([[1,0], [0, a]])
    x, hist = opt_diracs(D, n = n, lr = 0.1, max_it=1000)
    phi_gap = find_phi_gap(x)
    phis.append(torch.remainder(torch.pi/2 - phi_gap, torch.pi/2))
    
#%%
plt.plot(phis)


#%%
# plt.close('all')
# fig, ax = plt.subplots()
# ax.scatter(x[:, 0], x[:, 1], marker='.')
# circle = plt.Circle((0, 0), 1, color='g', clip_on=False, fill=False)
# ax.add_patch(circle)
# ax.plot([0, x_gap[0]], [0, x_gap[1]])