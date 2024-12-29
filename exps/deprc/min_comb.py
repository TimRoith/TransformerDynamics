import torch
from tfdy.optim_diracs import usa_flow, proj, OptimDiracs
import matplotlib.pyplot as plt
import numpy as np
#%%

n = 5
d = 100
kwargs = {
    'max_it': 500,
    'tau': 0.1,
    'n':n,
    'beta':1,
    'sgn': -1,
    'sigma':0.,
    'erg_int':1
}
