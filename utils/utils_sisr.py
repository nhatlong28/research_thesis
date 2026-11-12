import torch


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxHxW
        sf: split factor
    Returns:
        b: NxCx(H/sf)x(W/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    Zero-pad the PSF and correct its center before computing the FFT.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxW, complex tensor
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    return torch.fft.fftn(otf, dim=(-2, -1))


def upsample(x, sf):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxHxW
    '''
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., ::sf, ::sf].copy_(x)
    return z


def pre_calculate(y, k, sf):
    '''
    Args:
        y: NxCxHxW, LR input, [0, 1]
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    h, w = y.shape[-2:]
    FB = p2o(k, (h * sf, w * sf))
    FBC = torch.conj(FB)
    F2B = torch.abs(FB).square()
    FBFy = FBC * torch.fft.fftn(upsample(y, sf), dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def data_solution(x0, FB, FBC, F2B, FBFy, rho, sf):
    '''FFT solution of ||y - H(x)||^2 + rho * ||x - x0||^2
    Args:
        x0: NxCxHxW, HR input, [0, 1]
        FB, FBC, F2B, FBFy: from pre_calculate
        rho: positive scalar or Nx1x1x1 DC weight
        sf: integer

    Returns:
        NxCxHxW, HR solution without clipping
    '''
    FR = FBFy + torch.fft.fftn(rho * x0, dim=(-2, -1))
    FBR = torch.mean(splits(FB * FR, sf), dim=-1)
    invW = torch.mean(splits(F2B, sf), dim=-1)
    invWBR = FBR / (invW + rho)
    FX = (FR - FBC * invWBR.repeat(1, 1, sf, sf)) / rho
    return torch.fft.ifftn(FX, dim=(-2, -1)).real


def data_consistency(x0, FB, FBC, F2B, FBFy, rho, sf):
    '''data consistency with range conversion
    Args:
        x0: Nx3xHxW, HR input, [-1, 1]
        FB, FBC, F2B, FBFy: from LR input in [0, 1]
        rho: positive scalar or Nx1x1x1 DC weight
        sf: integer

    Returns:
        Nx3xHxW, corrected HR image in model scale, without clipping
    '''
    with torch.autocast(device_type=x0.device.type, enabled=False):
        x0 = x0.float() * 0.5 + 0.5
        x0 = data_solution(x0, FB, FBC, F2B, FBFy, rho, sf)
        return x0 * 2 - 1
