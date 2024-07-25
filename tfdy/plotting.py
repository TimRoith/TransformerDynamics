from matplotlib.patches import Ellipse

def add_ellipse(ax, m, w, h, **kwargs):
    ellipse = Ellipse(m, w, h, angle=0, fill=False, **kwargs)
    ax.add_patch(ellipse)
    return ellipse
    
def vis_optim(ax, opt):
    add_ellipse(ax, (0,0), 2,2, color='k')
    add_ellipse(ax, (0,0), 2/(opt.D[0,0]**0.5), 2/(opt.D[1,1]**0.5), color='red')
    x = opt.x()
    ax.scatter(x[:, 0], x[:, 1], marker='.')
    ax.axis('equal')