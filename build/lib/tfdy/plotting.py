from matplotlib.patches import Ellipse
import torch
from .utils import sph2cart
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
from sklearn.neighbors import BallTree


def add_ellipse(ax, m, w, h, **kwargs):
    ellipse = Ellipse(m, w, h, angle=0, fill=False, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def add_sphere(ax, n=50, **kwargs):
    p,t = (torch.linspace(0, 2* torch.pi, n), torch.linspace(0, torch.pi, n))
    pp,tt = torch.meshgrid(p,t, indexing='ij')
    XYZ = sph2cart(torch.stack([pp,tt],dim=-1))
    ax.plot_surface(XYZ[...,0], XYZ[...,1], XYZ[...,2], shade=False, **kwargs)
    
def vis_optim(ax, opt):
    add_ellipse(ax, (0,0), 2,2, color='k')
    add_ellipse(ax, (0,0), 2/(opt.D[0,0]**0.5), 2/(opt.D[1,1]**0.5), color='red')
    x = opt.x()
    ax.scatter(x[:, 0], x[:, 1], marker='.')
    ax.axis('equal')
    
    
def set_up_plots():
    rc('font',**{'family':'serif','sans-serif':['Helvetica'], 'size':22})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    
    
def fig_ax_3D(figsize=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def estimate_density(x,X):
    n = x.shape[0]
    tree = BallTree(x, leaf_size=2)
    h = 2*(4*torch.std(x)**5/(3*n))**(1/5)
    density = tree.kernel_density(X.flatten(end_dim=1), h=h, kernel='gaussian')
    density /= density.max()
    return density

def vis_sph_particles(ax, x, X, facecolors=None, title=None):
    ax.plot_surface(
        *[X[...,i] for i in range(3)],
        shade=False,
        edgecolor='none',
        facecolors=facecolors,
        alpha=0.2)
    ax.scatter(x[:,0], x[:,1], x[:,2], alpha=1, color='orange',marker='.')
    ax.axis('equal')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title(title)
