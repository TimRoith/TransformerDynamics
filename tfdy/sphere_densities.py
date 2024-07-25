# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:21:45 2024

@author: kabrisam
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

def surface_reg(surf_grid, surf_grid_shift):
    return (surf_grid_shift[0]-surf_grid[0])*(surf_grid[1] - surf_grid_shift[1])

def surface_trap(Dsurf_grid, Dsurf_grid_shift):
    return 0.5*(Dsurf_grid_shift[0]-Dsurf_grid[0])*\
        (Dsurf_grid[1] + Dsurf_grid[2] - Dsurf_grid_shift[1] - Dsurf_grid_shift[2])

def transform_points(p, D):
    Dp = np.matmul(D,p.reshape(3,-1)).reshape(p.shape)
    return (Dp/np.linalg.norm(Dp, axis = 0))

def cart2phi(p):
    return np.sign(p[1])*np.arccos(p[0]/np.sqrt(p[0]**2+p[1]**2)) + (p[1] < 0)*2*np.pi

    
    
#%%

N = 100
D = np.diag(np.array([1,1,1]))
#D = np.diag(1/np.diag(D))
phi = np.linspace(0, 2 * np.pi, N)
theta = np.linspace(0, np.pi, N)

a_grid = np.array(np.meshgrid(phi, theta))

p_grid = np.array([np.cos(a_grid[0])*np.sin(a_grid[1]),
                  np.sin(a_grid[0])*np.sin(a_grid[1]),
                  np.cos(a_grid[1])])

surf_grid = np.array([a_grid[0], p_grid[-1]])
surf_grid_shift = np.array([np.roll(a_grid[0], (0,-1), axis = 1),
                            np.roll(p_grid[-1], (-1,0), axis = 0)])


Dp_grid = transform_points(p_grid, D)

Dphi_grid = cart2phi(p_grid)
Dphi_grid[0] = Dphi_grid[1]

Dsurf_grid = np.array([Dphi_grid, Dp_grid[-1], np.roll(Dp_grid[-1], (0,-1), axis=1)])
Dsurf_grid_shift = np.array([np.roll(Dphi_grid, (0,-1), axis = 1),
                              np.roll(Dp_grid[-1], (-1,0), axis = 0),
                              np.roll(Dp_grid[-1], (-1,-1), axis = (0,1))])


old_surfs = surface_reg(surf_grid, surf_grid_shift)
new_surfs = surface_trap(Dsurf_grid, Dsurf_grid_shift)


plt.close('all')

faces = new_surfs/(4*np.pi*old_surfs)
faces = faces.round(decimals = 8)
#faces = old_surfs/(4*np.pi*new_surfs)
norm=colors.Normalize(vmin = np.min(faces),
                      vmax = np.max(faces), clip = False)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.plot_surface(Dp_grid[0], Dp_grid[1], Dp_grid[2], shade=False, facecolors=cm.coolwarm(norm(faces)))
ax.axis('equal')
fig.colorbar(cm.ScalarMappable(norm = norm, cmap=cm.coolwarm), ax=ax)
