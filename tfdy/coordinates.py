import torch



def sph2cart(phir, excl_r=True):
    if phir.ndim == 1: phir=phir[:,None] #1d case at_least2d doesn't give right result
    if excl_r:
        phi = phir
        r = 1
        out = torch.ones(phi.shape[:-1] + (phi.shape[-1] + 1,)) * r
    else:
        phi, r = (phir[..., :-1], phir[..., -1:])
        out = torch.ones_like(phir) * r
    out[..., :-1] *= torch.cos(phi)
    out[..., 1:]  *= torch.cumprod(torch.sin(phi), dim=-1)
    return out

def init_sph_normal(shape=None, excl_r=True):
    shape = (1,2) if shape is None else shape
    z = torch.randn(shape)
    z = z/torch.linalg.norm(z,dim=-1, keepdim=True)
    return cart2sph(z, excl_r=excl_r)

def cart2sph(x, excl_r=True):
    xx = torch.flip(x * x, dims=(-1,))
    cx = torch.flip(torch.cumsum(xx, dim=-1)**0.5, dims=(-1,))
    phi = torch.atan2(cx[..., 1:], x[..., :-1])
    phi[..., -1] *= torch.sign(x[..., -1])
    if excl_r:
        return phi
    else:
        return torch.cat([phi, xx.sum(axis=-1, keepdim=True)**0.5], dim=-1)