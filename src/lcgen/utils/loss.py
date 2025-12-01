import torch

def recon_loss(flux, flux_err, recon_flux):

    # Simple Gaussian NLL on mean+sigma (sigma = softplus(logs))
    # Simple modified chi squared loss
    obs_sigma = flux_err.abs().clamp(min=1e-6)
    var = obs_sigma ** 2
    res2 = (recon_flux - flux) ** 2
    nll = 0.5 * (torch.log(2.0 * torch.pi * var) + res2 / var)
    return nll