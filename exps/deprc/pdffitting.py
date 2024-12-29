import torch
import torch.nn as nn
from tfdy.utils import integral_scalar_prod, sph2cart, pol2cart
import matplotlib.pyplot as plt
import numpy as np
#%%
class poly_dist:
    def __init__(self, deg):
        self.deg = deg
        self.params = nn.Parameter(torch.ones((deg,)))
        
    def __call__(self, x):
        evalp = torch.clamp(torch.sum(x[:,None]**torch.arange(self.deg-1, -1, -1) * 
                         self.params, dim=-1), min=0)
        return evalp/torch.sum(evalp)*x.shape[0]
    
        
    def normalize_w(self,):
        self.params.data *= 1/self.params.max()
        
class exp_dist:
    def __init__(self, ):
        self.params = nn.Parameter(torch.ones((1,)))
        
    def __call__(self, x):
        evalp = 0.5 * torch.exp(-self.params * (x-torch.pi/2)**2) + 0.5 * torch.exp(-self.params * (x+torch.pi/2)**2)
        return evalp/torch.sum(evalp)#*x.shape[0]
    
        
    def normalize_w(self,):
        pass
        
class exp_guess:
    def __init__(self, eps=1):
        co = np.array([ 
                1.15873192e+05,  5.46912620e+04,  1.39741928e+04, -1.13194179e+04,
               -2.52650568e+04, -3.11225153e+04, -3.14659259e+04, -2.82961231e+04,
               -2.31375445e+04, -1.71216247e+04, -1.10584037e+04, -5.49786594e+03,
               -7.82341721e+02,  2.90885862e+03,  5.52155296e+03,  7.09464224e+03,
                7.73313158e+03,  7.58554061e+03,  6.82511979e+03,  5.63437352e+03,
                4.19246872e+03,  2.66517840e+03,  1.19707631e+03, -9.42394030e+01,
               -1.12205675e+03, -1.83270345e+03, -2.20600144e+03, -2.25403621e+03,
               -2.01822069e+03, -1.56461525e+03, -9.77473192e+02, -3.51030513e+02,
                2.20329744e+02,  6.53203744e+02,  8.86914712e+02,  8.94708686e+02,
                6.92520507e+02,  3.42549058e+02, -5.20320100e+01, -3.65402440e+02,
               -4.83702743e+02, -3.54761280e+02, -4.05444191e+01,  2.61881534e+02,
                2.94922329e+02, -2.14727493e+01, -2.74305085e+02,  1.53201899e+02,
               -2.81786781e+01,  3.99737141e+00, -1.28117088e-02])
        f = np.polyval(co, eps)
        f = np.tan(eps * (1 + (1/2)**0.5)) * 2**0.5
        c = 0.44536
        c = (2/3)**2
        f = np.sinh(np.tan(eps + c) - 0.46475) + eps
        #f = torch.exp(torch.pi/2*(eps/(1-eps))**0.5)
        
        
        self.params = nn.Parameter(f *  torch.ones(1))
    
    def __call__(self, theta):
        
        evalp = torch.exp(-self.params * torch.cos(2 * theta))
        return evalp/torch.sum(evalp)
    
    def normalize_w(self,):
        pass
    
    
class inv_cos:
    def __init__(self, eps=1):
        self.params = nn.Parameter(torch.ones(1))
    
    def __call__(self, theta):
        
        evalp = 1/(self.params + torch.cos(2 * theta))
        return evalp/torch.sum(evalp)
    
    def normalize_w(self,):
        pass
    
#%%
num_int_eval = 200

model = poly_dist(3)



class energy:
    def __init__(self, D, num_int_eval=500):
        self.D  = D
        self.Xp = torch.linspace(-torch.pi, torch.pi, num_int_eval)
        self.X  = pol2cart(self.Xp)
        self.num_int_eval = num_int_eval
     
    def __call__(self, prob):
        p = prob(self.Xp)
        p = p[:,None] * p[None,:]
        y = (self.D@self.X.T).T
        sp = torch.exp(torch.inner(self.X[:,None,:], y[None,...])).squeeze()
        return (sp * p).sum()


    
#%%
ps = []
es = torch.linspace(0,0.9,50)
for e in es:
    D = torch.diag(torch.tensor([1, 1-e], dtype=torch.float))
    model = exp_guess(eps=e)
    E = energy(D=D,num_int_eval=num_int_eval)
    opt = torch.optim.Adam([model.params], lr=0.1)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.95)
    
    print('----------')
    print('Starting with e: ' + str(e))
    print('Initial guess: ' + str(E(model).item()))
    for i in range(400):
        opt.zero_grad()
        l = E(model)
        l.backward()
        opt.step()
        sched.step(l)
        model.normalize_w()
        if i%100==0:
            print(l.item())
            
    ps.append(model.params)   
#%%

plt.close('all')
pa = torch.tensor([p.item() for p in ps])
co = np.polyfit(es,pa, deg=50)

g = lambda x: 2.2* x/((1-x**2)**0.5)
#g = lambda x: torch.pi/2*torch.atanh(x)
f = lambda x: torch.exp(g(x)) - 1#**0.5)
f = lambda x: x**2/((1-x)**2)
f = lambda x: torch.cosh(g(x)) -1
f = lambda x: torch.tan(x * (1 + (1/2)**0.5)) * 2**0.5
plt.plot(es,pa, marker='*')
plt.plot(es,f(es))
plt.plot(es,np.polyval(co, es))

#plt.figure()
#plt.bar(np.arange(len(co)), co)



#%%
Xp = torch.linspace(-torch.pi, torch.pi, num_int_eval)
plt.plot(Xp, model(Xp).detach())
#plt.ylim([0,2])
    