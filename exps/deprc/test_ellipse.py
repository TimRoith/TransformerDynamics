import torch
from elliptics import opt_diracs, find_phi_gap, pol2cart
import matplotlib.pyplot as plt

d1 = .5
d2 = 1
D = 1.* torch.tensor([[d1,0], [0, d2]])
n = 100
x, hist = opt_diracs(D, n = n, lr = 0.01, max_it=1000, a=1/(d1**0.5))