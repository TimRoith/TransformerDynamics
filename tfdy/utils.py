import torch
from scipy.cluster.vq import kmeans
import numpy as np
import math
from scipy.special import iv, kv, kn, yn
from tfdy.coordinates import sph2cart

class integral_scalar_prod:
    def __init__(self, D):
        self.D = D
     
    def scalar_prod(self, x, y):
        y = (self.D@y.T).T
        return torch.exp(torch.inner(x[:,None,:], y[None,...]))
    
    def __call__(self, d):
        return self.scalar_prod(d, d).sum()/(d.shape[0]**2)
    
class weighted_integral_scalar_prod:
    def __init__(self, D, X = None, Y = None):
        self.D = D
        if X is not None and Y is not None:
            self.precompute_XY_val(X, Y)
            self.precomp = True
        else:
            self.precomp = False
    
    def precompute_XY_val(self, X, Y):
        Y = (self.D@Y.T).T
        self.XY_val = torch.exp(torch.inner(X[:,None,:], Y[None,...])).squeeze()

    def scalar_prod_precomp(self, wx, wy):
        return self.XY_val * wx.T * wy
        
    def scalar_prod(self, x, y, wx, wy):
        y = (self.D@y.T).T
        return (
            torch.exp(torch.inner(x[:,None,:], y[None,...])).squeeze() *
            wx.T * wy
        )
    
    def __call__(self, w, d = None):
        if self.precomp:
            return self.scalar_prod_precomp(w, w).sum()
        
        return self.scalar_prod(d, d, w, w).sum()

def grid_sph2cart(p, t):
    return sph2cart(torch.stack([p, t], dim=-1))

def cart2pol(x):
    return torch.arctan2(x[..., 1], x[..., 0])

def init_phi(n, dim, mode='default'):
    if dim == 2:
        return init_phi_2D(n, mode=mode)
    elif dim == 3:
        return init_phi_3D(n, mode=mode)
    else:
        raise ValueError('Only supported for 2- or 3 dimensions!')
        
def grid_phi(n):
    s = int(n**0.5)
    p,t = (torch.linspace(0, torch.pi, s), torch.linspace(0, 2*torch.pi, s))
    return torch.meshgrid(p,t, indexing='ij')

def grid_x_sph(n):
    p, t = grid_phi(n)
    return grid_sph2cart(p, t)[..., [1,2,0]] # switch to xyz

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
        
        
def cc_kmeans(x, tol=0.2):
    cs, v = kmeans(x, 3)
    if cs.shape[0] > 1:
        diff = np.linalg.norm(cs[None, ...] - cs[:,None, :],axis=-1) > 0.1
        s = (diff).sum()//(cs.shape[0] - 1)
        idx = np.append(np.where(diff[:, 0])[0], 0)
    else: s = 1
    return {'cc': max(s - 1, 0), 'cs': cs[idx, :]}


def exp_pdf_guess(theta, eps=1, f=None):
    if f is None:
        f = 1/5 * eps**2 + torch.e/2 * eps

    evalp = torch.exp(f * torch.cos(2 * theta))
    return evalp/torch.sum(evalp, axis=-1, keepdims=True)

def sphere_area(n):
    '''
    Returns the area of the n-dimensional sphere
    '''

    return (2 * np.pi**((n + 1)/2)) /  math.gamma((n + 1)/2)
    

def calculate_C_tilde(n):
    return sphere_area(n - 2) / (n - 1)

def calculate_C3(n):
    f = calculate_C_tilde(n) / sphere_area(n - 1)
    #I = integral_ecos_sinn(n)
    I = approx_integral(lambda x: np.exp(np.cos(x)) * np.sin(x)**(n),
                        1000, [0, np.pi])
    return f * I

def calculate_C2(n):
    f = sphere_area(n - 2)/sphere_area(n - 1)
    I = approx_integral(lambda x: np.exp(np.cos(x)) * np.cos(x)**2 * np.sin(x)**(n-2),
                        1000, [0, np.pi])

    return f * I - calculate_C3(n)

def calculate_C1(n):
    f = sphere_area(n - 2)/sphere_area(n - 1)
    I = approx_integral(lambda x: np.exp(np.cos(x)) * np.cos(x) * np.sin(x)**(n-1),
                        1000, [0, np.pi])

    return f * I

def calculate_alpha(n):
    return -calculate_C1(n) / calculate_C2(n)

def calculate_beta_2D(M, N=1000):
    x = torch.linspace(0, 2 * torch.pi, N)
    x = sph2cart(x, excl_r=True)
    I = (x * (x @ M.T)).sum() / N
    return - calculate_alpha(2) * I

class perturb_density:
    def __init__(self, M, N):
        self.M = M
        self.alpha = calculate_alpha(M.shape[0])
        self.beta = calculate_beta_2D(M)
        self.dx = 1 / N

    def __call__(self, x, eps=1):
        return self.dx + eps * self.nu_star(x)
    def eval_sph_coord(self, phi, eps = 1):
        return self.__call__(sph2cart(phi, excl_r=True), eps=eps)
    
    def nu_star(self, x,):
        return (self.alpha * (x * (x @ self.M.T)).sum(-1) + self.beta) * self.dx
    
    def nu_star_sph(self, phi):
        return self.nu_star(sph2cart(phi, excl_r=True))

def approx_integral(fun, n, lims):
    x = np.linspace(lims[0], lims[1], n)
    dx = x[1] - x[0]
    return np.sum(fun(x)) * dx

def integral_ecos_sinn(n):
    m = (n-1)//2
    f = married_couple_seating(m)
    if n == 3:
        f2 = 4
    else:
        f2 = f * bessel_poly(m, 1)/loopless_linear_chord_diagram(m)
    return (f2/np.e - f * np.e) * (-1)**(m-1)

def married_couple_seating(n):
    '''
    Returns the number of ways to seat n married couples at a round table
    '''
    return (np.pi * 
     iv(n+1/2, 1) * (-1)**n + 
     kv(n+1/2, 1)
     ) * np.exp(-1) * (2/np.pi)**(1/2) * 2**n* math.factorial(n)

def num_fun_1(n):
    return np.exp(1-(1-n)**(1/2))

def bessel_poly(n, x):
    return (2/(np.pi * x)) ** (1/2) * np.exp(1/x) * kv(n+1/2, 1/x)

def loopless_linear_chord_diagram(n):
    if n < 0 or n == 1:
        return 0
    elif n == 0:
        return 1
    return (
        (2 * n - 1) * loopless_linear_chord_diagram(n - 1) + 
        loopless_linear_chord_diagram(n - 2)
    )

def return_fac(n):
    return (round(bessel_poly(n, 1)), loopless_linear_chord_diagram(n))
