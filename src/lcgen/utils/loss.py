import torch

def recon_loss(flux, flux_err, recon_flux):

    # Simple Gaussian NLL on mean+sigma (sigma = softplus(logs))
    # Simple modified chi squared loss
    obs_sigma = flux_err.abs().clamp(min=1e-6)
    var = obs_sigma ** 2
    res2 = (recon_flux - flux) ** 2
    nll = 0.5 * (torch.log(2.0 * torch.pi * var) + res2 / var)
    return nll


def bounded_horizon_future_nll(h_fwd, h_bwd, t_enc, model, flux, flux_err, mask=None, K: int = 128):
    """
    Compute the average NLL loss where each source hidden h_fwd[:, i] predicts up to K
    future targets j = i+1 .. i+K (bounded horizon). For each target j we average the
    NLLs from all valid predictions aimed at j, then average across all targets that
    received at least one valid prediction.

    Args:
        h_fwd: (B, L, H) tensor of forward hidden states (one per timestep)
        h_bwd: (B, L, H) tensor of backward hidden states (or None). If
               provided and the model is bidirectional, the loss will use both
               forward and backward hidden slices to form the same head input
               used at inference.
        t_enc: (B, L, Te) tensor of time-encodings per timestep
        model: the sequence model instance (used to access head, head_norm, and
           any time-conditioning modules such as `time_scale`). The
           function will use `model.gauss_head` as the prediction head and
           apply the same post-LN time-conditioning used in `model.forward` so
           training and inference match.
        flux: (B, L) ground-truth flux values
        flux_err: (B, L) measurement errors (used by recon_loss)
        mask: (B, L) optional mask where 1=observed, 0=masked. If provided,
              predictions are only computed for unmasked targets, and source
              hidden states must come from unmasked positions.
        K: maximum horizon (int)

    Returns:
        loss: scalar tensor (average NLL across targets that have >=1 valid prediction)
        stats: dict with keys 'total_preds' for diagnostics
        per_k_mean: dict mapping k -> mean NLL (float) for predictions at horizon k
    """
    device = h_fwd.device
    B, L, H = h_fwd.shape
    Te = t_enc.size(-1)

    flux = flux.to(device)
    flux_err = flux_err.to(device)

    # We'll compute a global average over ALL valid predictions.
    # Accumulate total NLL and total prediction count.
    max_k = min(K, L - 1)
    total_preds = 0
    total_sum_nll = torch.tensor(0.0, device=device)

    # Grab head and optional modules from model
    head = getattr(model, 'gauss_head', model)
    head_norm = getattr(model, 'head_norm', None)
    # time_scale expected to be a scalar nn.Parameter on the model; if absent
    # we simply won't scale the time slice.
    time_scale = getattr(model, 'time_scale', None)

    per_k_mean = {}
    for k in range(1, max_k + 1):
        # Symmetric bidirectional offset: predict targets at positions k to L-1-k
        # Forward hidden at j-k (info up to j-k-1), backward hidden at j+k (info from j+k+1)
        # This ensures neither direction sees any data within k steps of the target
        
        if 2*k >= L:
            # Not enough positions for symmetric offset
            continue
            
        # Target positions: k to L-1-k (inclusive)
        n_targets = L - 2*k
        
        # Time encodings at target positions
        t_tgt = t_enc[:, k:L-k, :]        # (B, n_targets, Te)

        # prepare inputs for head: concat along last dim. If the model is
        # bidirectional and backward hidden states were provided, build the
        # same fused input used in `model.forward` (src_f, src_b, t_tgt).
        if getattr(model, 'direction', None) == 'bi' and h_bwd is not None:
            # Forward hidden at positions 0 to L-2k-1 (predicting k to L-1-k)
            src_f = h_fwd[:, :n_targets, :]
            # Backward hidden at positions 2k to L-1 (info from 2k+1 to L)
            src_b = h_bwd[:, 2*k:, :]
            inputs_k = torch.cat([src_f, src_b, t_tgt], dim=-1)   # (B, n_targets, 2H+Te)
            flat_in = inputs_k.view(-1, 2 * H + Te)
        else:
            h_src = h_fwd[:, :n_targets, :]        # (B, n_targets, H)
            inputs_k = torch.cat([h_src, t_tgt], dim=-1)   # (B, n_targets, H+Te)
            flat_in = inputs_k.view(-1, H + Te)           # (B*n_targets, H+Te)

        # Apply head normalization and the same time-conditioning used in
        # `model.forward` before calling the head so training/inference match.
        if head_norm is not None:
            normed = head_norm(flat_in)
            if Te > 0:
                h_hidden = normed[:, :-Te]
                h_time = normed[:, -Te:]
                if time_scale is not None:
                    h_time = h_time * time_scale
                normed = torch.cat([h_hidden, h_time], dim=1)
        else:
            normed = flat_in

        # target ground truth and errors (symmetric: positions k to L-1-k)
        flux_tgt = flux[:, k:L-k]                 # (B, n_targets)
        ferr_tgt = flux_err[:, k:L-k]             # (B, n_targets)

        # Compute validity mask for predictions
        # A prediction is valid if:
        # 1. The target position is unmasked (we have ground truth to compare against)
        # 2. The forward source position (j-k) is unmasked
        # 3. The backward source position (j+k) is unmasked (for bidirectional)
        if mask is not None:
            mask_fwd_src = mask[:, :n_targets]     # (B, n_targets) - mask at forward source positions
            mask_tgt = mask[:, k:L-k]              # (B, n_targets) - mask at target positions
            if getattr(model, 'direction', None) == 'bi' and h_bwd is not None:
                mask_bwd_src = mask[:, 2*k:]       # (B, n_targets) - mask at backward source positions
                valid = (mask_fwd_src * mask_tgt * mask_bwd_src)
            else:
                valid = (mask_fwd_src * mask_tgt)
        else:
            valid = torch.ones_like(flux_tgt, device=device)

        # If the model exposes a conditional flow (zuko) named `flow`, use it
        # to compute per-prediction negative log-likelihoods. The flow will
        # be conditioned on the same normalized head input used in forward
        # plus the measurement error at the target timestep.
        flow_module = getattr(model, 'flow', None)
        if flow_module is not None:
            # Prepare context: normed is (B*n_targets, C); append ferr_tgt flattened
            ferr_flat = ferr_tgt.contiguous().view(-1)          # (B*n_targets,)
            ctx = torch.cat([normed, ferr_flat.unsqueeze(1)], dim=1)  # (N, C+1)

            # zuko flow call returns a Distribution-like object; compute log_prob
            dist = flow_module(ctx)
            # flux targets: shape (B, n_targets) -> flatten to (N, 1) for event dim 1
            flux_flat = flux_tgt.contiguous().view(-1, 1)
            logp = dist.log_prob(flux_flat)   # expected shape (N, 1) or (N,)
            # normalize shape to (N,)
            if logp.dim() > 1:
                logp = logp.view(-1)
            nll_flat = -logp
            nll_k = nll_flat.view(B, n_targets)
        else:
            # predict means (allow gradients to flow)
            preds_k = head(normed).view(B, n_targets)        # (B, n_targets)

            # compute NLL per prediction using existing recon_loss
            nll_k = recon_loss(flux_tgt, ferr_tgt, preds_k)   # (B, n_targets)

        # compute mean NLL for this k across valid predictions
        valid_count = int(valid.sum().item())
        if valid_count > 0:
            mean_nll_k = float((nll_k * valid).sum().item() / valid_count)
        else:
            mean_nll_k = float('nan')
        per_k_mean[k] = mean_nll_k

        # accumulate global sums
        total_sum_nll = total_sum_nll + (nll_k * valid).sum()
        total_preds += valid_count

    if total_preds == 0:
        return torch.tensor(0.0, device=device), {'total_preds': 0}

    loss = total_sum_nll / float(total_preds)
    return loss, {'total_preds': total_preds}, per_k_mean