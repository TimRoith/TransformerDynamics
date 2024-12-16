import torch

class integral_scalar_prod:
    def __init__(self, D):
        self.D = D
     
    def scalar_prod(self, x, y):
        y = (self.D@y.T).T
        return torch.exp(torch.inner(x[:,None,:], y[None,...]))
    
    def __call__(self, d):
        return self.scalar_prod(d, d).sum()/(d.shape[0]**2)
    
def pol2cart(phi, a=1.):
    return torch.stack([ a * torch.cos(phi), torch.sin(phi)]).T

def sph2cart(phi):
    return torch.stack([torch.sin(phi[..., 0]) * torch.cos(phi[..., 1]), 
                        torch.sin(phi[..., 0]) * torch.sin(phi[..., 1]),
                        torch.cos(phi[..., 0])], dim=-1)

def grid_sph2cart(p, t):
    return sph2cart(torch.stack([p, t], dim=-1))

def cart2pol(x):
    return torch.arctan2(x[..., 1], x[..., 0])

def cart2sph(x):
    return torch.stack([torch.acos(x[...,2]),  torch.arctan2(x[...,1], x[...,0])])

def init_phi(n, dim, mode='default'):
    if dim == 2:
        return init_phi_2D(n, mode=mode)
    elif dim == 3:
        return init_phi_3D(n, mode=mode)
    else:
        raise ValueError('Only supported for 2- or 3 dimensions!')
        
def grid_phi(n):
    s = int(n**0.5)
    p,t = (torch.linspace(0, 2* torch.pi, s), torch.linspace(0, torch.pi, s))
    return torch.meshgrid(p,t, indexing='ij')

def grid_x_sph(n):
    p, t = grid_phi(n)
    return grid_sph2cart(p, t)

def init_phi_3D(n, mode='sunflower'):
    if mode =='grid':
        pp,tt = grid_phi(n)
        return torch.stack([pp.ravel(), tt.ravel()], dim=1)
    elif mode == 'sunflower' or mode =='default':
        # this method is taken from here: 
        # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        # which offers more insights into the topic
        idx = torch.arange(0, n) + 0.5
        return torch.stack([torch.arccos(1 - 2*idx/n), 
                            torch.remainder(torch.pi * (1 + 5**0.5) * idx, 2*torch.pi)], dim=-1)
    else:
        raise ValueError('Unknown init mode: ' +str(mode))

def init_phi_2D(n, D = None, mode='default'):
    if mode == 'default':
        return torch.linspace(-torch.pi, torch.pi, n+1)[:-1]
    elif mode == 'estimate':
        if D is None:
            raise ValueError('Estimate init requires the problem matrix D')
        elif abs(D[0,0] - 1) > 1e-5:
            raise ValueError('Estimate is currently implemented for d1 != 0')
        phi = torch.linspace(-torch.pi, torch.pi, n+1)[:-1]
        xproj = torch.cat([s * pol2cart(torch.atan(torch.tan(phi) * (D[0,0]**(3/2)))) for s in [1,-1]])
        xproj = xproj[torch.where(xproj.abs()[:,1]>0.01)]
        return cart2pol(xproj)
    else:
        raise ValueError('Unknown init mode: ' +str(mode))