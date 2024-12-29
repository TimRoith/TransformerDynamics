import torch
import tfdy
from tfdy.optim_diracs import usa_flow
from tfdy.utils import pol2cart, init_phi_2D
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
#%%
D = torch.diag(torch.tensor([1., -1], dtype=torch.float))
Xp = init_phi_2D(n=100) + 0.1
X = pol2cart(Xp)
x , hist = usa_flow(D, n = 500, max_it=10000, tau=0.5, x=X, sigma=0.)

#%%

plt.close('all')
fig, ax = plt.subplots()
ax.plot(*[torch.cat([X[:,i], X[0:1,i]]) for i in [0,1]], zorder=-1)
ax.scatter(x[:,0], x[:, 1], color='xkcd:sky', s=15)
ax.axis('square')

plt.tight_layout(pad=0, w_pad=0,h_pad=0.)
plt.savefig('results/circ_sol_D=' + str(D.numpy()).replace('\n','').replace(' ','') + '.png')