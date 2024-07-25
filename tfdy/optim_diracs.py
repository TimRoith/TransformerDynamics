import torch
import torch.nn as nn

from .utils import integral_scalar_prod, pol2cart, cart2pol


class OptimDiracs2D:
    def __init__(self, D=None,):
        self.D = torch.eye(2) if D is None else D
        self.loss_fct = integral_scalar_prod(D)
        
    def loss(self, phi):
        return self.loss_fct(self.phi2cart(phi))

    def optimize(
        self, n=10, max_it = 500, a=1., 
        phi = None, sgn=1., sigma=0., opt_kwargs=None
    ):

        self.init_phi(phi=phi, n=n)
        opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        self.init_opt(**opt_kwargs)
       
        print('Starting with loss value: ' +str(self.cur_loss().item()))
        self.run_opt_loop(max_it=max_it, sgn=sgn, sigma=sigma)
            
        x_ret = self.x()
        print('Finished with loss value: ' + str(self.cur_loss().item()))
        return x_ret
    
    def run_opt_loop(self, max_it=10, sgn=1., sigma=0.):
        self.hist = []
        phi_best = None
        phi_best_val = sgn * float('inf')
        
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
        self.phi = phi_best

    def init_phi(self, phi=None, n=10):
        phi = torch.zeros((n,)).uniform_(-torch.pi, torch.pi) if phi is None else phi
        self.phi = nn.Parameter(phi)
        
    def init_opt(self, **kwargs):
        self.opt = torch.optim.Adam([self.phi], **kwargs)
        #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50)
        
    def phi2cart(self, phi):
        return pol2cart(phi)
        
    def cart2phi(self, x):
        return cart2pol(x)
    
    def x(self,):
        return self.phi2cart(self.phi)
    
    def cur_loss(self,):
        return self.loss(self.phi)
        
        