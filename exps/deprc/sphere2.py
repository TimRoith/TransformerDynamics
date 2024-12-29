import tfdy
import tfdy.plotting as tplt
from tfdy.utils import integral_scalar_prod, sph2cart
import torch
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib import cm
#%%
d1 = 1.
d2 = .0
d3 = 0.5
beta = 5.
tau = .1

n = 300
D = torch.diag(torch.tensor([d1, d2, d3], dtype=torch.float))
print(D)
#%%
def proj(x):
    return x/torch.linalg.vector_norm(x, dim=-1,keepdims=True)

def P(x, y):
    return y - (y*x).sum(-1)[:,None] * x

class vel:
    def __init__(self, D = None, beta:float = 1):
        self.beta = beta * torch.ones((1,1))
        self.D = torch.eye(2) if D is None else D
    
    def __call__(self, x):
        g = torch.exp(self.beta * (x@(self.D@x.T)))
        d = x.shape[0]#torch.sum(g, dim=-1)[...,None]
        z = (1/d) * torch.sum(g[..., None] * x[None,...], dim=-2)
        return z@self.D
    
#%%
beta=2.
tau=0.1

v = vel(D=D, beta=beta)
E = integral_scalar_prod(D=D)

x = proj(torch.randn((n,3)))

for i in range(300):
    x = proj(x - tau * v(x))

    if i%100 == 0:
        print(E(x))
            


#%%
plt.close('all')
#tplt.set_up_plots()
fig, ax = tplt.fig_ax_3D(figsize=(15,15))


X = tfdy.utils.grid_x_sph(10000)

density = tplt.estimate_density(x, X)

tplt.vis_sph_particles(
    ax, x=x, X=X,
    facecolors=cm.coolwarm(density.reshape(X.shape[:2] + (-1,))),
    #title = 'Loss Value: ' + str(optim.cur_loss().item())
)


#%%
fig, ax = plt.subplots(1)
P = tfdy.utils.cart2sph(X)
p = tfdy.utils.cart2sph(x)

ax.imshow(density.reshape(100,100))

#%%
name = (
        'results/' + 
        reduce(lambda a,b: a+'-'+b, [str(D[i,i].item()) for i in range(3)]) + 
        '.png'
        )
plt.tight_layout()
plt.savefig(name)
