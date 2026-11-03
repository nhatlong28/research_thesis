import torch


def splits(a, sf):
    '''Split the frequency tensor into sf x sf distinct blocks.

    Args:
        a: NxCxHxW frequency tensor; H and W must be divisible by sf.
        sf: integer scale factor.

    Returns:
        b: NxCx(H/sf)x(W/sf)x(sf^2) tensor of grouped frequency blocks.
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''Convert a centered point-spread function to an optical transfer function.

    Zero-pad the PSF, shift its center to the origin, and take its FFT.
    This represents circular convolution on the requested spatial grid.

    Args:
        psf: NxCxhxw centered spatial blur kernel.
        shape: target spatial size (H, W).

    Returns:
        otf: complex NxCxHxW frequency response.
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    return torch.fft.fftn(otf, dim=(-2, -1))


def upsample(x, sf):
    '''Upsample by inserting zeros, the adjoint of phase-zero decimation.

    Args:
        x: NxCxHxW LR tensor.
        sf: integer scale factor.

    Returns:
        z: NxCx(H*sf)x(W*sf) tensor, with x at indices [::sf, ::sf].
    '''
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., ::sf, ::sf].copy_(x)
    return z


def pre_calculate(y, k, sf):
    '''Precompute kernel and measurement terms reused by the DC iterations.

    Args:
        y: NxCxHxW observed LR tensor on the [0,1] scale.
        k: centered blur kernel, typically 1x1xhxw, shared across RGB/batch.
        sf: integer scale factor.

    Returns:
        FB: FFT of the kernel on the HR grid.
        FBC: complex conjugate of FB.
        F2B: squared magnitude of FB.
        FBFy: FFT of the adjoint-blurred, zero-inserted LR measurement.
        These tensors are reused at every sampling timestep.
    '''
    h, w = y.shape[-2:]
    FB = p2o(k, (h * sf, w * sf))
    FBC = torch.conj(FB)
    F2B = torch.abs(FB).square()
    FBFy = FBC * torch.fft.fftn(upsample(y, sf), dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def data_solution(x0, FB, FBC, F2B, FBFy, rho, sf):
    '''Solve the quadratic super-resolution data subproblem using FFT.

    Minimize ||y - S(k*x)||^2 + rho * ||x - x0||^2, where convolution
    is circular and S keeps indices [::sf, ::sf]. Both norms use sums.

    Args:
        x0: NxCxHxW clean HR prediction on the [0,1] scale.
        FB, FBC, F2B, FBFy: frequency terms returned by pre_calculate().
        rho: positive regularization weight, constant across timesteps here.
        sf: integer scale factor.

    Returns:
        HR solution with the same shape as x0. The unconstrained solution
        is not clipped and may contain values outside [0,1].
    '''
    FR = FBFy + torch.fft.fftn(rho * x0, dim=(-2, -1))
    FBR = torch.mean(splits(FB * FR, sf), dim=-1)
    invW = torch.mean(splits(F2B, sf), dim=-1)
    invWBR = FBR / (invW + rho)
    FX = (FR - FBC * invWBR.repeat(1, 1, sf, sf)) / rho
    return torch.fft.ifftn(FX, dim=(-2, -1)).real


def data_consistency(x0, FB, FBC, F2B, FBFy, rho, sf):
    '''Apply DiffPIR Step 4 to a UPSR prediction before PixelUnshuffle.

    Args:
        x0: Nx3xHxW RGB HR prediction in the model's [-1,1] scale.
        FB, FBC, F2B, FBFy: precomputed terms from LR y on the [0,1] scale.
        rho: positive constant DC weight.
        sf: integer scale factor.

    Returns:
        Corrected RGB HR prediction in the model's scale, with no clipping.
        FFT calculations use float32 inputs with autocast disabled.
    '''
    with torch.autocast(device_type=x0.device.type, enabled=False):
        x0 = x0.float() * 0.5 + 0.5
        x0 = data_solution(x0, FB, FBC, F2B, FBFy, rho, sf)
        return x0 * 2 - 1
