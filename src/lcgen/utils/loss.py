import torch

def recon_loss(flux, flux_err, recon_flux):

    # Simple Gaussian NLL on mean+sigma (sigma = softplus(logs))
    # Simple modified chi squared loss
    obs_sigma = flux_err.abs().clamp(min=1e-6)
    var = obs_sigma ** 2
    res2 = (recon_flux - flux) ** 2
    nll = 0.5 * (torch.log(2.0 * torch.pi * var) + res2 / var)
    return nll


def bounded_horizon_future_nll(h_fwd, t_enc, head, flux, flux_err, K: int = 32):
    """
    Compute the average NLL loss where each source hidden h_fwd[:, i] predicts up to K
    future targets j = i+1 .. i+K (bounded horizon). For each target j we average the
    NLLs from all valid predictions aimed at j, then average across all targets that
    received at least one valid prediction.

    Args:
        h_fwd: (B, L, H) tensor of forward hidden states (one per timestep)
        t_enc: (B, L, Te) tensor of time-encodings per timestep
        head: callable / nn.Module that maps concatenated [h_src, t_enc_tgt]
              of shape (N, H+Te) -> (N, 1) predicted mean
        flux: (B, L) ground-truth flux values
        flux_err: (B, L) measurement errors (used by recon_loss)
        mask: (B, L) boolean or {0,1} mask where 1=observed (valid source/target)
        K: maximum horizon (int)

    Returns:
        loss: scalar tensor (average NLL across targets that have >=1 valid prediction)
        stats: dict with keys 'total_preds' and 'targets_with_preds' for diagnostics
    """
    device = h_fwd.device
    B, L, H = h_fwd.shape
    Te = t_enc.size(-1)

    flux = flux.to(device)
    flux_err = flux_err.to(device)

    # accumulators per-target
    sum_nll_per_target = torch.zeros(B, L, device=device)
    count_per_target = torch.zeros(B, L, device=device)

    max_k = min(K, L - 1)
    total_preds = 0

    for k in range(1, max_k + 1):
        src_slice = slice(0, L - k)
        tgt_slice = slice(k, L)

        h_src = h_fwd[:, src_slice, :]        # (B, L-k, H)
        t_tgt = t_enc[:, tgt_slice, :]        # (B, L-k, Te)

        # prepare inputs for head: concat along last dim
        inputs_k = torch.cat([h_src, t_tgt], dim=-1)   # (B, L-k, H+Te)
        flat_in = inputs_k.view(-1, H + Te)           # (B*(L-k), H+Te)

        # predict means
        with torch.no_grad():
            # head may contain parameters; however predictions are used to compute loss
            # we call head without torch.no_grad so grads propagate. Remove no_grad.
            pass
        preds_k = head(flat_in).view(B, L - k)        # (B, L-k)

        # target ground truth and errors
        flux_tgt = flux[:, tgt_slice]                 # (B, L-k)
        ferr_tgt = flux_err[:, tgt_slice]             # (B, L-k)

        # When no masking is used, every source/target pair in this diagonal is valid
        valid = torch.ones_like(flux_tgt, device=device)

        # compute NLL per prediction using existing recon_loss
        nll_k = recon_loss(flux_tgt, ferr_tgt, preds_k)   # (B, L-k)

        # accumulate (include all pairs)
        sum_nll_per_target[:, tgt_slice] += nll_k * valid
        count_per_target[:, tgt_slice] += valid
        total_preds += int(valid.sum().item())

    # Only include targets with at least one prediction
    targets_with_preds = (count_per_target > 0)
    num_targets_with_preds = int(targets_with_preds.sum().item())

    if num_targets_with_preds == 0:
        # no supervision available
        return torch.tensor(0.0, device=device), {'total_preds': total_preds, 'targets_with_preds': 0}

    avg_nll_per_target = torch.zeros_like(sum_nll_per_target)
    avg_nll_per_target[targets_with_preds] = (
        sum_nll_per_target[targets_with_preds] / count_per_target[targets_with_preds]
    )

    # final loss: mean across all (b, t) that have >=1 prediction
    loss = avg_nll_per_target[targets_with_preds].mean()

    return loss, {'total_preds': total_preds, 'targets_with_preds': num_targets_with_preds}