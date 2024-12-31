from matplotlib.patches import Ellipse
import torch
from tfdy.coordinates import sph2cart, cart2sph
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
from sklearn.neighbors import BallTree, KernelDensity
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from warnings import warn
from tfdy.utils import grid_x_sph, init_phi_2D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


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

def estimate_density(x,X, h=None):
    n = x.shape[0]
    tree = BallTree(x, leaf_size=2)
    
    if X.ndim > 2:
        X = X.flatten(end_dim=1)

    kde = KernelDensity(kernel='gaussian', bandwidth='scott' if h is None else h).fit(x)
    #density = tree.kernel_density(X, h='scott', kernel='gaussian')
    log_density = kde.score_samples(X)
    #log_density /= log_density.max()
    return log_density

def vis_sph_particles(ax, x=None, X=None, facecolors=None, title=None,):
    if X is not None:
        ax.plot_surface(
            *[X[...,i] for i in range(3)],
            shade=False,
            edgecolor='none',
            facecolors=facecolors,
            alpha=0.2)
    if x is not None:
        ax.scatter(x[:,0], x[:,1], x[:,2], alpha=1, color='orange',marker='.')
    ax.axis('equal')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.set_title(title)


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
    
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    return make_colormap([low, c('white'), 0.5, c('white'), high])

def get_surf_colormap():
    return make_colormap([(1.,1.,1.), (0.2745, 0.2549, 0.5882)]).reversed()
    #return diverge_map(low=(1.,1.,0.9), high=(0.2745, 0.2549, 0.5882))

def plot_sphere(ax, X=None, **kwargs):
    X = grid_x_sph(50**2) if X is None else X
    dkwargs = {'shade':False, 
               'edgecolor':'k', 
               'rstride':1,
               'cstride':1,
               'linewidth':0.1}
    dkwargs.update(kwargs)
    
    ax.plot_surface(
        *[X[...,i] for i in range(3)],
        **dkwargs
        )
    
def lazy_view_idx(x, ax):
    az, el = (ax.azim/180)*torch.pi, (ax.elev/180)*torch.pi,
    phi = cart2sph(x[..., [2,0,1]], excl_r=True)
    idx  = torch.where((phi[...,1] <  -az/2) * (phi[...,1] > (-torch.pi-az/2)) *
                       (phi[...,0] < torch.pi - el)
                       +
                       (phi[...,0] < el)
                       )[0]
    idx_c = torch.ones(x.shape[0])
    idx_c[idx] = False
    return idx, torch.where(idx_c)[0]
    
def plot_circle(ax, **kwargs):
    c = plt.Circle((0, 0), 1., **kwargs)
    ax.add_patch(c)
    
def plot_colored_circle(ax, D,  **kwargs):
    phi = init_phi_2D(100)
    X = sph2cart(phi, excl_r=True)
    X = torch.cat([X,X[0:1, :]])
    c = (X*(X@D)).sum(-1)
    colored_line(X[:,0].numpy(), X[:,1].numpy(), c, ax, **kwargs)
    
def setup_style():
    try:
        import scienceplots
        plt.style.use(['science'])
    except:
        warn('Scienplots not available!')

    
class PlotConf:
    def __init__(self, figsize=(5,5)):
        setup_style()
        self.fig = plt.figure(figsize=figsize)
        self.axs = []
        self.cmap = get_surf_colormap()
   
    def save(self, name=None):
        plt.pause(0.3)
        fname = input('Please specify the filename: \n') if name is None else name
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.savefig('./results/' + fname + '.png', dpi=600)
        
class PlotConf2D(PlotConf):
    def add_ax(self, n, computed_zorder=False, lims=None):
        self.axs.append(self.fig.add_subplot(n))
        self.axs[-1].axis('square')
        for i in range(2): 
            pre = 'set_' + chr(120 + i)
            getattr(self.axs[-1], pre + 'ticks')(torch.linspace(-1,1,5))
            getattr(self.axs[-1], pre + 'lim')(([-1., 1.] if lims is None else lims))
            
    def init_axs(self, n, **kwargs):
        for i in range(n): self.add_ax(100 + 10 * n + (i+1), **kwargs)
    
    def plot_circle(self, idx=0, **kwargs):
        plot_circle(self.axs[idx], **kwargs)
    
    def plot_colored_circle(self, D, idx=0, vmin=0., vmax=1, **kwargs):
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        plot_colored_circle(self.axs[idx], D, cmap=self.cmap, norm=norm, **kwargs)
        

class PlotConf3D(PlotConf):
    
    def add_ax(self, n, computed_zorder=False, ticks=True, show_ticks = False, 
               labelpad = -10, num_ticks = 3, axlabelfontsize=25):
        self.axs.append(self.fig.add_subplot(n, 
                                             projection='3d', 
                                             computed_zorder=computed_zorder
                                             )
                        )
        self.axs[-1].axis('square')
        for i in range(3): 
            pre = 'set_' + chr(120 + i)
            if ticks:
                getattr(self.axs[-1], pre + 'ticks')(torch.linspace(-1,1,num_ticks))
            else:
                getattr(self.axs[-1], pre + 'ticks')([])

            if not show_ticks:
                getattr(self.axs[-1], pre + 'ticklabels')('')


            getattr(self.axs[-1], pre + 'label')('$z_' + str(i + 1) + '$', 
                                                 fontsize=axlabelfontsize, labelpad=labelpad)
            getattr(self.axs[-1], pre + 'lim')([-1., 1.])

        self.axs[-1].minorticks_off()
        
        
    def init_axs(self, n, **kwargs):
        for i in range(n): self.add_ax(100 + 10 * n + (i+1), **kwargs)
        
    def plot_sphere(self, idx=0, **kwargs):
        plot_sphere(self.axs[idx], **kwargs)
        
    def add_colorbar(self, idx=0, vmin=0, vmax=1, labelsize=18, 
                     **kwargs):
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=self.axs[idx], **kwargs)
        cbar.ax.tick_params(labelsize=labelsize) 
        return cbar

def fill_between_3d(ax,x1,y1,z1,x2,y2,z2,mode=1,c='steelblue',alpha=0.6, zorder=0):
    
    """
    
    Function similar to the matplotlib.pyplot.fill_between function but 
    for 3D plots.
       
    input:
        
        ax -> The axis where the function will plot.
        
        x1 -> 1D array. x coordinates of the first line.
        y1 -> 1D array. y coordinates of the first line.
        z1 -> 1D array. z coordinates of the first line.
        
        x2 -> 1D array. x coordinates of the second line.
        y2 -> 1D array. y coordinates of the second line.
        z2 -> 1D array. z coordinates of the second line.
    
    modes:

        mode = 1 -> Fill between the lines using the shortest distance between 
                    both. Makes a lot of single trapezoids in the diagonals 
                    between lines and then adds them into a single collection.
                    
        mode = 2 -> Uses the lines as the edges of one only 3d polygon.
           
    Other parameters (for matplotlib): 
        
        c -> the color of the polygon collection.
        alpha -> transparency of the polygon collection.
        
    Copied from:
    https://github.com/artmenlope/matplotlib-fill_between-in-3D/blob/master/FillBetween3d.py
    """

    if mode == 1:
        
        for i in range(len(x1)-1):
            
            verts = [(x1[i],y1[i],z1[i]), (x1[i+1],y1[i+1],z1[i+1])] + \
                    [(x2[i+1],y2[i+1],z2[i+1]), (x2[i],y2[i],z2[i])]
            
            ax.add_collection3d(Poly3DCollection([verts],
                                                 alpha=alpha,
                                                 linewidths=0,
                                                 color=c,
                                                 zorder=zorder), 
                                                 )

    if mode == 2:
        
        verts = [(x1[i],y1[i],z1[i]) for i in range(len(x1))] + \
                [(x2[i],y2[i],z2[i]) for i in range(len(x2))]
                
        ax.add_collection3d(Poly3DCollection([verts],alpha=alpha,color=c))
            