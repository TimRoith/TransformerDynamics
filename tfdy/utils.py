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
                        torch.cos(phi[..., 0])])

def cart2pol(x):
    return torch.arctan2(x[..., 1], x[..., 0])

def cart2sph(x):
    return torch.stack([torch.acos(x[...,2]),  torch.arctan2(x[...,1], x[...,0])])

def init_phi_2D(n):
    s = int(n**2)
    p,t = (torch.linspace(0, 2* torch.pi))

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