import torch
import torch.nn as nn

from .utils import integral_scalar_prod, init_phi, weighted_integral_scalar_prod
from .coordinates import sph2cart, cart2sph, init_sph_normal


def proj(x):
    return x/torch.linalg.vector_norm(x, dim=-1,keepdim=True)

def proj_tang(x, y):
    return y - (y*x).sum(-1)[:,None] * x

@torch.no_grad()
def find_decr_step_size(D, x, tau_max = 10, sgn = 1., max_it = 100, fac=0.5):
    n = x.shape[0]
    E = integral_scalar_prod(D)
    erg_init = E(x)
    erg_new = torch.inf
    tau = tau_max
    
    for i in range(max_it):
        x_new = x - sgn * (tau/n) * torch.exp((x @ (D @ x.T))) @ x @ D.T
        erg_new = E(x_new)
        if sgn * erg_new < fac * sgn * erg_init:
            break
        tau *= 0.9
    return tau
        
    

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

class ExponantiatedGD(torch.optim.Optimizer): 
    def __init__(self, params, lr=0.1): 
        super().__init__(params, defaults={'lr': lr}) 
        self.state = dict() 
        for group in self.param_groups: 
            for p in group['params']: 
                self.state[p] = dict() 
      
    # Step Method 
    def step(self): 
        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue

                v = p.data * torch.exp(-group['lr'] * p.grad.data)
                p.data = v / v.sum()

def optimizer_density_2D(D, N=1000, max_it = 500, opt_kwargs=None):
    X = sph2cart(init_phi(N, 2), excl_r=True)
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs
    weights = nn.Parameter(torch.ones(N,1)/N)
    E = weighted_integral_scalar_prod(D, X = X, Y = X)
    opt = ExponantiatedGD([weights], **opt_kwargs)
    hist = []
    for i in range(max_it):
        opt.zero_grad()
        loss = E(weights)
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return weights, hist


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
        
        