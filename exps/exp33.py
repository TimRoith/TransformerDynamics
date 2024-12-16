import torch

class g:
    def __init__(self, d=1):
        self.d = d
    def __call__(self, th, ph, p):
        d = self.d
        ct, cp, st, sp = (torch.cos(th)[:, None], torch.cos(ph)[None, :], 
                          torch.sin(th)[:, None], torch.sin(ph)[None, :])

        exp = torch.exp
        p, = ((exp(z)/exp(z).sum())[:, None] for z in (p,))
    
        # out = (
        # (
        # (exp(d * ct * cp) + exp(-d * ct * cp)) * 
        # (exp(    st * sp) - exp(-    st * sp)) * 
        #     st * cp
        # -
        # (exp(d * ct * cp) - exp(-d * ct * cp)) * 
        # (exp(    st * sp) + exp(-    st * sp)) *
        # d * ct * sp
        # ) * p
        #
        # +
        # #
        # (
        # (exp(d * st * cp) + exp(-d * st * cp)) * 
        # (exp(    ct * sp) - exp(-    ct * sp)) *
        #     ct * cp
        # -
        # (exp(d * st * cp) - exp(-d * st * cp)) *
        # (exp(    ct * sp) + exp(-    ct * sp)) * 
        # d * st * sp
        # ) * ps
        #)
        # out = -(
        #     exp( d * ct * cp + st * sp) + 
        #     exp( d * ct * cp - st * sp) -
        #     exp(-d * ct * cp + st * sp) -
        #     exp(-d * ct * cp - st * sp)
        # ) * d * ct * sp
        # out += (
        #     exp( d * ct * cp + st * sp) - 
        #     exp( d * ct * cp - st * sp) +
        #     exp(-d * ct * cp + st * sp) -
        #     exp(-d * ct * cp - st * sp)
        # )     * st * cp
        
        # out *= p
        
        out = -exp( d * ct * cp + st * sp) * (d * ct * sp - st * cp) * p
        
        return out
        
#%%
num = 2000
th = torch.linspace(0, 2*torch.pi, num)
ph = torch.linspace(0, 2*torch.pi, 100)
p = torch.nn.Parameter(torch.ones(size=(num,)))

G = g(d=1.0)


opt = torch.optim.Adam([p,], lr=0.04)
best = torch.cat([p,]).detach().clone()
rmin = ((G(th, ph, p)).sum(axis=0)**2).sum()
#best = torch.cat([p, torch.flip(ps, dims=(0,))]).clone().detach()
better_found = False
print('Starting with: ' +str(rmin.item()))
#%%
for n in range(500):
    opt.zero_grad()
    r = ((G(th, ph, p)).sum(axis=0)**2).sum()
    if r < rmin:
        best = torch.cat([
            p, 
            #torch.flip(
            #ps, 
            #dims=(0,))
            ]).clone().detach()
        better_found = True
        rmin = r
    if r < 1e-10:
        break
    
    r.backward()
    opt.step()
    
    if n%100 == 0 or better_found:
        print(r)
        better_found = False
    
#%%
import matplotlib.pyplot as plt
plt.close('all')
best = ((torch.exp(best)/torch.exp(best).sum()))

plt.plot(th, best)


#%%
# import numpy as np
# from pysr import PySRRegressor
# model = PySRRegressor(
#     niterations=100,  # < Increase me for better results
#     binary_operators=["+", "*", '-', '/','^'],
#     constraints={'^':(-1,1)},
#     populations=25,
#     unary_operators=[
#         "exp",
#         "sin",
#         "cos",
#         "tan",
#         #"inv(x) = 1/x",
#         "abs",
#         "cosh",
#         "sinh",
#         "tanh",
#         "atan",
#         "asinh",
#         "sqrt",
#         "log",
#         #"acosh",
#         "atanh_clip",
#         #"square",
#         #"cube",
#         # ^ Custom operator (julia syntax)
#     ],
#     warm_start=False,
#     maxsize=10,
#     extra_sympy_mappings={"inv": lambda x: 1 / x, 
#                           "oadd": lambda x: 1+x},
#     # ^ Define operator for SymPy as well
# )

#%%
#
model.fit(th[:,None], best[:,None])
