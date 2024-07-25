import torch
import torch.nn as nn
import scipy as sp
import scipy.optimize
import numpy as np
    
   
def pol2cart(phi, a=1.):
    return torch.stack([ a * torch.cos(phi), torch.sin(phi)]).T

def cart2pol(x):
    return torch.arctan2(x[...,1], x[...,0])

def find_phi_gap(x):
    phi, _ = torch.sort(cart2pol(x))
    diff = (phi[1:] - phi[:-1]).abs()
    idx = torch.argmax(diff)
    return torch.remainder(phi[idx-1], torch.pi/2)


def opt_diracs(D=None, n=10, max_it = 500, lr=0.001, a=1., phi = None, sgn=1., 
               sigma=0.):
    D = torch.eye(2) if D is None else D
    phi = torch.zeros((n,)).uniform_(-torch.pi, torch.pi) if phi is None else phi
    phi = nn.Parameter(phi)
    opt = torch.optim.Adam([phi], lr=lr)
    #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50)
    loss_fct = integral_scalar_prod(D)

    print('Starting with loss value: ' +str(loss_fct(pol2cart(phi,a=a)).item()))
    hist = []
    phi_best = None
    phi_best_val = sgn * float('inf')
    
    for i in range(max_it):
        opt.zero_grad()
        phi.data += torch.zeros_like(phi.data).uniform_(-sigma, sigma)
        loss = sgn * loss_fct(pol2cart(phi, a=a))
        loss.backward()
        opt.step()
        #sched.step(loss)
        hist.append(loss.item())
        if loss.item() < phi_best_val:
            phi_best = phi.data.clone()
            phi_best_val =  loss.item()
        
    x_ret = pol2cart(phi_best,a=a).detach()
    print('Finished with loss value: ' + str(loss_fct(x_ret).item()))
    return x_ret, hist


def angles_in_ellipse(
        num,
        a,
        b):
    assert(num > 0)
    assert(a < b)
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e2 = (1.0 - a ** 2.0 / b ** 2.0)
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(
            lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles)
        angles = res.x 
    return angles

def get_uniform_ellipse(a=1., b=1., n = 100):
    phi = angles_in_ellipse(n, a, b)
    e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
    arcs = sp.special.ellipeinc(phi, e)
    return np.stack([b * np.sin(phi), a * np.cos(phi)]).T

