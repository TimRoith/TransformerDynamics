import torch
import torch.nn as nn

from .utils import integral_scalar_prod, pol2cart, cart2pol, init_phi
from .coordinates import sph2cart, cart2sph, init_sph_normal


def proj(x):
    return x/torch.linalg.vector_norm(x, dim=-1,keepdim=True)

def proj_tang(x, y):
    return y - (y*x).sum(-1)[:,None] * x

@torch.no_grad()
def usa_flow(
        D, 
        beta:float = 1., n:int = 100, tau:float = 0.05, 
        max_it:int = 1000, erg_int:int=50, 
        x = None, sigma = 0.01, sgn=1,
        track_fct=None):
    x = proj(torch.randn(n, D.shape[0])) if x is None else x.clone()
    E = integral_scalar_prod(D=D)
    hist = []
    for i in range(max_it):
        #x -= proj_tang(x, sgn * (tau/n) * torch.exp(beta * (x @ (D @ x.T))) @ x @ D.T)
        x -= sgn * (tau/n) * torch.exp(beta * (x @ (D @ x.T))) @ x @ D.T
        x += sigma * torch.normal(0,1, x.shape)
        x /= torch.linalg.vector_norm(x, dim=-1,keepdim=True)
        
        sigma *=0.9
        if i%erg_int == 0:
            Ex = E(x)
            hist.append(Ex)
            if track_fct is not None: track_fct(x)
        
    return x, hist


class OptimDiracs:
    def __init__(self, D=None,):
        self.D = torch.eye(2) if D is None else D
        self.dim = D.shape[-1]
        self.loss_fct = integral_scalar_prod(D)
        
    def loss(self, phi):
        return self.loss_fct(sph2cart(phi, excl_r=True))

    def optimize(
        self, n=10, max_it = 500, a=1., 
        phi = None, sgn=1., sigma=0., opt_kwargs=None
    ):

        self.init_phi(phi=phi, n=n)
        opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        self.init_opt(**opt_kwargs)
       
        print('Starting with loss value: ' +str(sgn * self.cur_loss().item()))
        self.run_opt_loop(max_it=max_it, sgn=sgn, sigma=sigma)
            
        x_ret = self.x()
        print('Finished with loss value: ' + str(sgn * self.cur_loss().item()))
        return x_ret
    
    def run_opt_loop(self, max_it=10, sgn=1., sigma=0.):
        self.hist = []
        phi_best = self.phi.data.clone()
        phi_best_val = float('inf')
        
        for _ in range(max_it):
            self.opt.zero_grad()
            self.phi.data += torch.zeros_like(self.phi.data).uniform_(-sigma, sigma)
            loss = sgn * self.loss(self.phi)
            loss.backward()
            self.opt.step()
            #sched.step(loss)
            self.hist.append(loss.item())
            if loss.item() < phi_best_val:
                phi_best = self.phi.data.clone()
                phi_best_val =  loss.item()
                
            sigma*=0.9
        self.phi = phi_best

    def init_phi(self, phi=None, n=10):
        phi = init_sph_normal(shape=(n, self.dim), excl_r=True) if phi is None else phi
        self.phi = nn.Parameter(phi)
        
    def init_opt(self, **kwargs):
        self.opt = torch.optim.SGD([self.phi], **kwargs)
        #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50)

    def x(self,):
        return sph2cart(self.phi, excl_r=True)
    
    def cur_loss(self,):
        return self.loss(self.phi)
        
        