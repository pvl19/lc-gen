"""K-fold cross-validation for age prediction from latent space + BPRP0.

This script trains k separate models, each on (k-1)/k of the data, and generates
age predictions for the held-out fold. The result is an unbiased age prediction
for every star, made by a model that never saw that star during training.

Star aggregation modes (--star_aggregation):
  none             - One sample per LC/sector; kfold splits by sector (legacy, has leakage
                     for multi-sector stars). Use only for backward compatibility.
  predict_mean     - One sample per LC/sector; kfold splits by *star* (no leakage);
                     per-sector predictions are averaged per star before reporting metrics.
  latent_mean      - Per-sector latent vectors are mean-pooled per star before kfold.
  latent_median    - Per-sector latent vectors are median-pooled per star before kfold.
  cross_sector     - Hidden states from all sectors of a star are concatenated before
                     multiscale pooling, yielding one latent vector per star.

Usage:
    python scripts/kfold_age_inference.py \\
        --model_path output/performance_tests/cf_final/model.pt \\
        --output_dir output/age_predictor/kfold \\
        --star_aggregation cross_sector \\
        --n_folds 10
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
    import zuko
except ImportError:
    zuko = None

# Ensure local 'src' is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

# ---------------------------------------------------------------------------
# Age grid and outlier model constants (mirrors ChronoFlow convention)
# ---------------------------------------------------------------------------
PRIOR_LOGA_MYR = (0.0, 4.14)          # log10(1 Myr) → log10(~14 Gyr)
LOGA_GRID_DEFAULT_SIZE = 1000
LOGA_GRID = np.linspace(PRIOR_LOGA_MYR[0], PRIOR_LOGA_MYR[1], LOGA_GRID_DEFAULT_SIZE)
P_CLUSTER_MEM = 0.9                    # assumed cluster membership probability
P_OUTLIER     = 0.05                   # outlier fraction per star


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky 2015).

    Identity on the forward pass; multiplies the gradient by -lambda on the
    backward pass. Placed between the bottleneck and a sector classifier so that
    minimising the classifier's loss pushes the *encoder* to make the bottleneck
    sector-INVARIANT (while the classifier itself still learns to read sector).
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class AgePredictor(nn.Module):
    """Neural Likelihood Estimation via NSF:
    models p(z_pca | log10_age, BPRP0, log10(BPRP0_err), log10(MG_quick)).

    No learned encoder — the input z is a fixed PCA projection of the autoencoder latent,
    fit per fold on training data only (no leakage, no collapse).

    Training: flow conditioned on (log10_age, bprp0, log10(bprp0_err), log10(mg)), trained
    to assign high likelihood to the observed PCA-compressed latent z.

    Inference: likelihood evaluated over a dense age grid; posterior = likelihood × uniform prior.
    """

    def __init__(self, pca_dim: int, flow_transforms: int = 8,
                 flow_hidden_features: list = None, use_mg: bool = False):
        super().__init__()
        if zuko is None:
            raise ImportError("zuko is required for the flow head. Install with: pip install zuko")
        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        self.pca_dim = pca_dim
        self.use_mg  = use_mg
        context_dim  = 4 if use_mg else 3
        self.flow = zuko.flows.NSF(
            features=pca_dim,
            context=context_dim,
            transforms=flow_transforms,
            hidden_features=flow_hidden_features,
        )

    def _per_sample_nll(self, log_prob_flow: torch.Tensor, z: torch.Tensor,
                         mem_prob: torch.Tensor = None) -> torch.Tensor:
        """Per-sample NLL under the flow + static-outlier mixture (no reduction)."""
        if mem_prob is None:
            p_mem = P_CLUSTER_MEM
        else:
            p_mem = torch.where(torch.isnan(mem_prob), torch.full_like(mem_prob, P_CLUSTER_MEM), mem_prob)
        nf_weight    = p_mem * (1.0 - P_OUTLIER)
        ln_p_outlier = (
            -0.5 * z.pow(2).sum(dim=-1)
            - 0.5 * self.pca_dim * np.log(2.0 * np.pi)
        )
        ln_p_combined = torch.logsumexp(
            torch.stack([
                torch.log(nf_weight)       + log_prob_flow,
                torch.log(1.0 - nf_weight) + ln_p_outlier,
            ], dim=0), dim=0
        )
        return -ln_p_combined  # (N,)

    def _nll_with_outlier(self, log_prob_flow: torch.Tensor, z: torch.Tensor,
                          mem_prob: torch.Tensor = None) -> torch.Tensor:
        return self._per_sample_nll(log_prob_flow, z, mem_prob).mean()

    def forward(self, z: torch.Tensor, log_age: torch.Tensor,
                bprp0: torch.Tensor, log_bprp0_err: torch.Tensor,
                log_mg: torch.Tensor = None, mem_prob: torch.Tensor = None) -> torch.Tensor:
        # Multi-sample ages: log_age shape (B, K) → per-star mean then per-batch mean.
        # Single-sample (legacy): log_age shape (B,).
        if log_age.dim() == 2:
            B, K = log_age.shape
            log_age_f = log_age.reshape(B * K)
            bprp0_f   = bprp0.unsqueeze(1).expand(B, K).reshape(B * K)
            berr_f    = log_bprp0_err.unsqueeze(1).expand(B, K).reshape(B * K)
            mg_f      = (log_mg.unsqueeze(1).expand(B, K).reshape(B * K)
                         if log_mg is not None else None)
            z_f       = z.unsqueeze(1).expand(B, K, -1).reshape(B * K, z.shape[-1])
            mp_f      = (mem_prob.unsqueeze(1).expand(B, K).reshape(B * K)
                         if mem_prob is not None else None)
            parts = [log_age_f, bprp0_f, berr_f]
            if self.use_mg:
                parts.append(mg_f)
            context = torch.stack(parts, dim=1)
            per_sample = self._per_sample_nll(self.flow(context).log_prob(z_f), z_f, mp_f)
            return per_sample.reshape(B, K).mean(dim=1).mean()

        parts = [log_age, bprp0, log_bprp0_err]
        if self.use_mg:
            parts.append(log_mg)
        context = torch.stack(parts, dim=1)
        return self._nll_with_outlier(self.flow(context).log_prob(z), z, mem_prob)

    def predict_stats(self, z: torch.Tensor, bprp0: torch.Tensor,
                      log_bprp0_err: torch.Tensor, log_mg: torch.Tensor,
                      loga_grid: torch.Tensor) -> dict:
        """Grid-based posterior: p(age | z, bprp0) ∝ p(z | age, bprp0) · p(age)."""
        B, D = z.shape
        G    = loga_grid.shape[0]

        z_exp     = z.unsqueeze(1).expand(B, G, D).reshape(B * G, D)
        bprp0_exp = bprp0.unsqueeze(1).expand(B, G).reshape(B * G)
        berr_exp  = log_bprp0_err.unsqueeze(1).expand(B, G).reshape(B * G)
        loga_exp  = loga_grid.unsqueeze(0).expand(B, G).reshape(B * G)
        parts     = [loga_exp, bprp0_exp, berr_exp]
        if self.use_mg:
            parts.append(log_mg.unsqueeze(1).expand(B, G).reshape(B * G))
        context   = torch.stack(parts, dim=1)

        log_lik   = self.flow(context).log_prob(z_exp).reshape(B, G)
        log_post  = log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)
        posterior = log_post.exp()

        map_idx = posterior.argmax(dim=1)
        map_est = loga_grid[map_idx]
        mean    = (posterior * loga_grid.unsqueeze(0)).sum(dim=1)
        cdf     = posterior.cumsum(dim=1)

        def pct(p: float) -> torch.Tensor:
            return loga_grid[(cdf >= p).float().argmax(dim=1)]

        return {
            'median': pct(0.5),
            'mean':   mean,
            'map':    map_est,
            'p16':    pct(0.16),
            'p84':    pct(0.84),
        }


class AgePredictorMLP(nn.Module):
    """NLE with a learned MLP encoder to compress the autoencoder latent.

    Architecture:
        latent (input_dim) → MLP encoder → bottleneck (bottleneck_dim, LayerNorm)
                                                   ↓
                NSF flow: p(bottleneck | log10_age, bprp0, log10(bprp0_err), log10(mg))

    Collapse prevention: an auxiliary linear head predicts log10_age from the bottleneck
    with an L1 loss scaled by aux_loss_weight. This gives the encoder a direct age gradient
    so it cannot ignore age-relevant dimensions.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int,
                 mlp_hidden: list = None,
                 flow_transforms: int = 8,
                 flow_hidden_features: list = None,
                 aux_loss_weight: float = 1.0,
                 use_mg: bool = False,
                 dropout: float = 0.1,
                 variance_reg_weight: float = 0.0,
                 n_sectors: int = 0,
                 adv_hidden: int = 64):
        super().__init__()
        if zuko is None:
            raise ImportError("zuko is required for the flow head. Install with: pip install zuko")
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        self.bottleneck_dim      = bottleneck_dim
        self.aux_loss_weight     = aux_loss_weight
        self.variance_reg_weight = variance_reg_weight
        self.use_mg              = use_mg
        context_dim              = 4 if use_mg else 3

        # MLP encoder
        layers = []
        prev = input_dim
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, bottleneck_dim), nn.LayerNorm(bottleneck_dim)]
        self.encoder = nn.Sequential(*layers)

        # Auxiliary age head (linear, trained with L1 loss)
        self.aux_age_head = nn.Linear(bottleneck_dim, 1)

        # NSF flow
        self.flow = zuko.flows.NSF(
            features=bottleneck_dim,
            context=context_dim,
            transforms=flow_transforms,
            hidden_features=flow_hidden_features,
        )

        # Optional gradient-reversal sector adversary on the bottleneck. Trained to
        # predict sector from z; the GRL flips its gradient into the encoder, pushing
        # z toward sector-invariance while the flow still fits age. Discarded at
        # inference (predict_stats never touches it).
        self.n_sectors = n_sectors
        if n_sectors > 0:
            self.sector_adversary = nn.Sequential(
                nn.Linear(bottleneck_dim, adv_hidden), nn.ReLU(),
                nn.Linear(adv_hidden, n_sectors),
            )
        else:
            self.sector_adversary = None

    def _per_sample_nll(self, log_prob_flow: torch.Tensor, z: torch.Tensor,
                         mem_prob: torch.Tensor = None) -> torch.Tensor:
        """Per-sample NLL under the flow + static-outlier mixture (no reduction)."""
        if mem_prob is None:
            p_mem = P_CLUSTER_MEM
        else:
            p_mem = torch.where(torch.isnan(mem_prob), torch.full_like(mem_prob, P_CLUSTER_MEM), mem_prob)
        nf_weight    = p_mem * (1.0 - P_OUTLIER)
        ln_p_outlier = (
            -0.5 * z.pow(2).sum(dim=-1)
            - 0.5 * self.bottleneck_dim * np.log(2.0 * np.pi)
        )
        ln_p_combined = torch.logsumexp(
            torch.stack([
                torch.log(nf_weight)       + log_prob_flow,
                torch.log(1.0 - nf_weight) + ln_p_outlier,
            ], dim=0), dim=0
        )
        return -ln_p_combined  # (N,)

    def _nll_with_outlier(self, log_prob_flow: torch.Tensor, z: torch.Tensor,
                          mem_prob: torch.Tensor = None) -> torch.Tensor:
        return self._per_sample_nll(log_prob_flow, z, mem_prob).mean()

    def _flow_nll(self, z: torch.Tensor, log_age: torch.Tensor,
                  bprp0: torch.Tensor, log_bprp0_err: torch.Tensor,
                  log_mg: torch.Tensor = None, mem_prob: torch.Tensor = None) -> torch.Tensor:
        """Flow NLL with optional per-star multi-sample averaging.

        log_age (B,)    → standard mean over batch (legacy path).
        log_age (B, K)  → expand other inputs over K, compute per-sample NLL,
                          then mean over K per star, then mean over B stars.
                          Avoids implicitly weighting by sample count.
        """
        if log_age.dim() == 2:
            B, K = log_age.shape
            log_age_f = log_age.reshape(B * K)
            bprp0_f   = bprp0.unsqueeze(1).expand(B, K).reshape(B * K)
            berr_f    = log_bprp0_err.unsqueeze(1).expand(B, K).reshape(B * K)
            mg_f      = (log_mg.unsqueeze(1).expand(B, K).reshape(B * K)
                         if log_mg is not None else None)
            z_f       = z.unsqueeze(1).expand(B, K, -1).reshape(B * K, z.shape[-1])
            mp_f      = (mem_prob.unsqueeze(1).expand(B, K).reshape(B * K)
                         if mem_prob is not None else None)
            parts = [log_age_f, bprp0_f, berr_f]
            if self.use_mg:
                parts.append(mg_f)
            context = torch.stack(parts, dim=1)
            per_sample = self._per_sample_nll(self.flow(context).log_prob(z_f), z_f, mp_f)
            return per_sample.reshape(B, K).mean(dim=1).mean()

        parts = [log_age, bprp0, log_bprp0_err]
        if self.use_mg:
            parts.append(log_mg)
        context = torch.stack(parts, dim=1)
        return self._nll_with_outlier(self.flow(context).log_prob(z), z, mem_prob)

    def _adversary_loss(self, z: torch.Tensor, sector_idx: torch.Tensor,
                        adv_lambda: float) -> torch.Tensor:
        """Cross-entropy of the GRL sector adversary (0 if disabled)."""
        if (self.sector_adversary is None or sector_idx is None or adv_lambda <= 0):
            return z.new_zeros(())
        logits = self.sector_adversary(GradReverse.apply(z, adv_lambda))
        return nn.functional.cross_entropy(logits, sector_idx)

    def forward(self, x: torch.Tensor, log_age: torch.Tensor,
                bprp0: torch.Tensor, log_bprp0_err: torch.Tensor,
                log_mg: torch.Tensor = None, mem_prob: torch.Tensor = None,
                loss_mode: str = 'full',
                sector_idx: torch.Tensor = None, adv_lambda: float = 0.0) -> torch.Tensor:
        """Compute loss.

        loss_mode:
            'full'     — NLL + aux L1 + variance reg (default, joint training)
            'aux_only' — aux L1 + variance reg only (encoder pre-training, flow frozen)
            'nll_only' — NLL only (flow training, encoder frozen)

        log_age may be (B,) or (B, K) — see _flow_nll. The aux head is a point
        estimator, so it always trains against the per-star central age (mean
        over K when multi-sample); per-sample MC on the aux head would just
        add noise without changing the L1 minimizer in expectation.

        sector_idx / adv_lambda: when the GRL adversary is enabled, add its
        cross-entropy to the encoder-training losses ('full', 'aux_only'). Skipped
        for 'nll_only' (the encoder is frozen there, so the GRL push is moot).
        """
        z         = self.encoder(x)
        aux_target = log_age.mean(dim=1) if log_age.dim() == 2 else log_age

        if loss_mode == 'aux_only':
            aux  = torch.mean(torch.abs(self.aux_age_head(z).squeeze(1) - aux_target))
            loss = aux  # weight is irrelevant when aux is the only loss term
            loss = loss + self._adversary_loss(z, sector_idx, adv_lambda)
        elif loss_mode == 'nll_only':
            return self._flow_nll(z, log_age, bprp0, log_bprp0_err, log_mg, mem_prob)
        else:  # 'full'
            nll  = self._flow_nll(z, log_age, bprp0, log_bprp0_err, log_mg, mem_prob)
            aux  = torch.mean(torch.abs(self.aux_age_head(z).squeeze(1) - aux_target))
            # Scale aux to match NLL magnitude so aux_loss_weight controls relative
            # contribution (1.0 = equal gradient magnitude) regardless of raw scales
            nll_scale = nll.detach().abs().clamp(min=1e-4)
            aux_scale = aux.detach().abs().clamp(min=1e-4)
            loss = nll + self.aux_loss_weight * (nll_scale / aux_scale) * aux
            loss = loss + self._adversary_loss(z, sector_idx, adv_lambda)

        if self.variance_reg_weight > 0 and z.shape[0] > 1:
            std      = z.std(dim=0)
            var_loss = torch.relu(1.0 - std).mean()
            loss     = loss + self.variance_reg_weight * var_loss
        return loss

    def predict_stats(self, x: torch.Tensor, bprp0: torch.Tensor,
                      log_bprp0_err: torch.Tensor, log_mg: torch.Tensor,
                      loga_grid: torch.Tensor) -> dict:
        """Grid-based posterior (same interface as AgePredictor.predict_stats)."""
        z = self.encoder(x)
        B, D = z.shape
        G    = loga_grid.shape[0]

        z_exp     = z.unsqueeze(1).expand(B, G, D).reshape(B * G, D)
        bprp0_exp = bprp0.unsqueeze(1).expand(B, G).reshape(B * G)
        berr_exp  = log_bprp0_err.unsqueeze(1).expand(B, G).reshape(B * G)
        loga_exp  = loga_grid.unsqueeze(0).expand(B, G).reshape(B * G)
        parts     = [loga_exp, bprp0_exp, berr_exp]
        if self.use_mg:
            parts.append(log_mg.unsqueeze(1).expand(B, G).reshape(B * G))
        context   = torch.stack(parts, dim=1)

        log_lik   = self.flow(context).log_prob(z_exp).reshape(B, G)
        log_post  = log_lik - torch.logsumexp(log_lik, dim=1, keepdim=True)
        posterior = log_post.exp()

        map_idx = posterior.argmax(dim=1)
        map_est = loga_grid[map_idx]
        mean    = (posterior * loga_grid.unsqueeze(0)).sum(dim=1)
        cdf     = posterior.cumsum(dim=1)

        def pct(p: float) -> torch.Tensor:
            return loga_grid[(cdf >= p).float().argmax(dim=1)]

        return {
            'median': pct(0.5),
            'mean':   mean,
            'map':    map_est,
            'p16':    pct(0.16),
            'p84':    pct(0.84),
        }


class AgePredictorNPE(nn.Module):
    """Neural Posterior Estimation: models p(log10_age | bottleneck, BPRP0,
    log10(BPRP0_err), [log10(MG)]).

    Inverts the NLE construction in AgePredictorMLP — the flow output is a 1D
    distribution over age, conditioned on the latent bottleneck *and* the colour
    features. This avoids modelling p(z | …) over a high-dim latent and instead
    directly amortises the posterior, which is what we ultimately use for
    inference anyway.

    Architecture:
        latent (input_dim) → MLP encoder → bottleneck (LayerNorm) ┐
                                                                  │
        context = [bottleneck ∥ bprp0 ∥ log_bprp0_err [∥ log_mg]] ┘
                                            │
                                            ▼
                            NSF flow → p(log10_age | context)

    Per-star outlier mixture: a uniform distribution over the configured age
    grid plays the same role as the standard-normal background in NLE — it
    protects the flow from being pulled by stars whose archive age is wrong
    (rather than just noisy; archive σ already enters via Gaussian samples).
    """

    def __init__(self, input_dim: int, bottleneck_dim: int,
                 mlp_hidden: list = None,
                 flow_transforms: int = 8,
                 flow_hidden_features: list = None,
                 aux_loss_weight: float = 1.0,
                 use_mg: bool = False,
                 dropout: float = 0.1,
                 variance_reg_weight: float = 0.0,
                 age_grid_range: tuple = PRIOR_LOGA_MYR):
        super().__init__()
        if zuko is None:
            raise ImportError("zuko is required for the flow head. Install with: pip install zuko")
        if mlp_hidden is None:
            mlp_hidden = [256, 128]
        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        self.bottleneck_dim      = bottleneck_dim
        self.aux_loss_weight     = aux_loss_weight
        self.variance_reg_weight = variance_reg_weight
        self.use_mg              = use_mg
        self.register_buffer('_grid_min',
                             torch.tensor(float(age_grid_range[0]), dtype=torch.float32))
        self.register_buffer('_grid_max',
                             torch.tensor(float(age_grid_range[1]), dtype=torch.float32))

        n_color     = 3 if use_mg else 2  # bprp0, log_bprp0_err [, log_mg]
        context_dim = bottleneck_dim + n_color

        # MLP encoder (mirrors AgePredictorMLP)
        layers = []
        prev = input_dim
        for h in mlp_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, bottleneck_dim), nn.LayerNorm(bottleneck_dim)]
        self.encoder = nn.Sequential(*layers)

        # Auxiliary linear age head — same role as in AgePredictorMLP: gives the
        # encoder a direct age gradient during stage 1 / aux-only mode so the
        # bottleneck cannot collapse into age-irrelevant directions.
        self.aux_age_head = nn.Linear(bottleneck_dim, 1)

        # 1D NSF flow over log10_age, conditioned on (bottleneck ∥ colours)
        self.flow = zuko.flows.NSF(
            features=1,
            context=context_dim,
            transforms=flow_transforms,
            hidden_features=flow_hidden_features,
        )

    def _build_context(self, z: torch.Tensor, bprp0: torch.Tensor,
                        log_bprp0_err: torch.Tensor,
                        log_mg: torch.Tensor = None) -> torch.Tensor:
        parts = [z, bprp0.unsqueeze(-1), log_bprp0_err.unsqueeze(-1)]
        if self.use_mg:
            parts.append(log_mg.unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def _per_sample_nll(self, log_prob_flow: torch.Tensor,
                         mem_prob: torch.Tensor = None) -> torch.Tensor:
        """Per-sample NLL under the flow + uniform-over-grid outlier mixture."""
        if mem_prob is None:
            p_mem = P_CLUSTER_MEM
        else:
            p_mem = torch.where(torch.isnan(mem_prob),
                                torch.full_like(mem_prob, P_CLUSTER_MEM), mem_prob)
        nf_weight    = p_mem * (1.0 - P_OUTLIER)
        ln_p_outlier = -torch.log(self._grid_max - self._grid_min)
        ln_p_outlier = ln_p_outlier.expand_as(log_prob_flow)
        ln_p_combined = torch.logsumexp(
            torch.stack([
                torch.log(nf_weight)       + log_prob_flow,
                torch.log(1.0 - nf_weight) + ln_p_outlier,
            ], dim=0), dim=0
        )
        return -ln_p_combined  # (N,)

    def _flow_nll(self, z: torch.Tensor, log_age: torch.Tensor,
                  bprp0: torch.Tensor, log_bprp0_err: torch.Tensor,
                  log_mg: torch.Tensor = None,
                  mem_prob: torch.Tensor = None) -> torch.Tensor:
        """Flow NLL with optional per-star multi-sample averaging.

        log_age (B,)    → standard mean over batch.
        log_age (B, K)  → expand context over K samples, mean per star, then per batch.
        """
        if log_age.dim() == 2:
            B, K   = log_age.shape
            ctx    = self._build_context(z, bprp0, log_bprp0_err, log_mg)         # (B, C)
            ctx_f  = ctx.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            target = log_age.reshape(B * K, 1)
            mp_f   = (mem_prob.unsqueeze(1).expand(B, K).reshape(B * K)
                      if mem_prob is not None else None)
            log_prob   = self.flow(ctx_f).log_prob(target)
            per_sample = self._per_sample_nll(log_prob, mp_f)
            return per_sample.reshape(B, K).mean(dim=1).mean()

        ctx      = self._build_context(z, bprp0, log_bprp0_err, log_mg)
        target   = log_age.unsqueeze(-1)
        log_prob = self.flow(ctx).log_prob(target)
        return self._per_sample_nll(log_prob, mem_prob).mean()

    def forward(self, x: torch.Tensor, log_age: torch.Tensor,
                bprp0: torch.Tensor, log_bprp0_err: torch.Tensor,
                log_mg: torch.Tensor = None, mem_prob: torch.Tensor = None,
                loss_mode: str = 'full') -> torch.Tensor:
        z          = self.encoder(x)
        aux_target = log_age.mean(dim=1) if log_age.dim() == 2 else log_age

        if loss_mode == 'aux_only':
            aux  = torch.mean(torch.abs(self.aux_age_head(z).squeeze(1) - aux_target))
            loss = aux
        elif loss_mode == 'nll_only':
            return self._flow_nll(z, log_age, bprp0, log_bprp0_err, log_mg, mem_prob)
        else:  # 'full'
            nll  = self._flow_nll(z, log_age, bprp0, log_bprp0_err, log_mg, mem_prob)
            aux  = torch.mean(torch.abs(self.aux_age_head(z).squeeze(1) - aux_target))
            nll_scale = nll.detach().abs().clamp(min=1e-4)
            aux_scale = aux.detach().abs().clamp(min=1e-4)
            loss = nll + self.aux_loss_weight * (nll_scale / aux_scale) * aux

        if self.variance_reg_weight > 0 and z.shape[0] > 1:
            std      = z.std(dim=0)
            var_loss = torch.relu(1.0 - std).mean()
            loss     = loss + self.variance_reg_weight * var_loss
        return loss

    def predict_stats(self, x: torch.Tensor, bprp0: torch.Tensor,
                      log_bprp0_err: torch.Tensor, log_mg: torch.Tensor,
                      loga_grid: torch.Tensor) -> dict:
        """Posterior over log10_age via direct flow evaluation.

        Returns dict with median, mean, map, p16, p84 — same interface as the
        NLE predictors so downstream batching code is unchanged.
        """
        z   = self.encoder(x)
        B   = z.shape[0]
        G   = loga_grid.shape[0]
        ctx = self._build_context(z, bprp0, log_bprp0_err, log_mg)            # (B, C)
        C   = ctx.shape[-1]

        ctx_exp  = ctx.unsqueeze(1).expand(B, G, C).reshape(B * G, C)
        grid_exp = loga_grid.unsqueeze(0).expand(B, G).reshape(B * G, 1)

        log_prob   = self.flow(ctx_exp).log_prob(grid_exp).reshape(B, G)
        log_post   = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
        posterior  = log_post.exp()

        map_idx = posterior.argmax(dim=1)
        map_est = loga_grid[map_idx]
        mean    = (posterior * loga_grid.unsqueeze(0)).sum(dim=1)
        cdf     = posterior.cumsum(dim=1)

        def pct(p: float) -> torch.Tensor:
            return loga_grid[(cdf >= p).float().argmax(dim=1)]

        return {
            'median': pct(0.5),
            'mean':   mean,
            'map':    map_est,
            'p16':    pct(0.16),
            'p84':    pct(0.84),
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _concat_sectors_with_time(h_list, t_list):
    """Stitch per-sector hidden states / times into a single continuous trajectory.

    Each sector's stored time array starts at 0 (build_pretrain_h5 resets
    `time -= time[0]` per sector). To run the time-aware pool on a cross-sector
    concatenation we need a monotonically increasing global time axis. We
    offset each subsequent sector by the previous sector's t_max plus the
    median within-sector cadence of that sector, so the inter-sector boundary
    looks like one regular cadence step rather than a discontinuity. The real
    multi-month TESS gap between sectors is *not* represented — pooling
    treats the concatenated stream as if observations were contiguous, which
    matches what the original sample-indexed pool implicitly did.
    """
    t_parts = []
    offset = 0.0
    for t in t_list:
        if t.numel() == 0:
            t_parts.append(t)
            continue
        t_shifted = t - t[0] + offset
        t_parts.append(t_shifted)
        if t.numel() > 1:
            dt_med = (t[1:] - t[:-1]).median().item()
        else:
            dt_med = 0.0
        offset = float(t_shifted[-1].item()) + max(dt_med, 1e-6)
    return torch.cat(h_list, dim=0), torch.cat(t_parts, dim=0)


def extract_latents_by_star(model, h5_path: str, device: torch.device,
                             sample_indices: np.ndarray, tic_ids_sorted: np.ndarray,
                             max_length: int = None, pooling_mode: str = 'multiscale',
                             batch_size: int = 32,
                             use_metadata: bool = False, use_conv_channels: bool = False,
                             compute_cross_sector: bool = True):
    """Single-pass extraction of per-sector AND cross-sector latents.

    Stars are accumulated into forward-pass batches of ~batch_size sectors so
    the GPU stays well-utilised.  After each forward pass the hidden states are
    split back into per-star groups; per-sector latents are computed immediately
    and the hidden states for each star are concatenated to produce the
    cross-sector latent.  Hidden states are discarded as soon as both latent
    types for a star are produced — peak memory is O(batch_size sectors), not
    O(total sectors).

    Args:
        model:             trained BiDirectionalMinGRU (eval mode expected)
        h5_path:           path to H5 data file
        device:            torch device
        sample_indices:    H5 indices in sorted order (same ordering as tic_ids_sorted)
        tic_ids_sorted:    TIC IDs aligned to sample_indices
        max_length:        per-sector max sequence length
        pooling_mode:      'mean' | 'multiscale' | 'final' — applied per sector
        batch_size:        target number of sectors per forward pass (stars are kept
                           whole, so actual batch size may slightly exceed this)
        use_metadata:      pass stellar metadata to model if available
        use_conv_channels: pass conv channels to model if available

    Returns:
        per_sector_latents:   (N, D) — one row per sector, same order as sample_indices
        cross_sector_latents: (S, D) — one row per unique star
        unique_tic_ids:       (S,)   — TIC ordering for cross_sector_latents
    """
    from plot_umap_latent import compute_multiscale_features
    from lcgen.models.TimeSeriesDataset import METADATA_FEATURES

    load_metadata = use_metadata and (model.meta_encoder is not None)
    N = len(sample_indices)

    # Load only lightweight arrays upfront (lengths + metadata are small scalars)
    with h5py.File(h5_path, 'r') as f:
        lengths = f['length'][sample_indices]

        if load_metadata and 'metadata' in f:
            from lcgen.models.MetadataAgePredictor import MetadataStandardizer
            meta_grp = f['metadata']
            raw = {}
            for feat in METADATA_FEATURES:
                if feat in meta_grp:
                    vals = meta_grp[feat][sample_indices]
                    if vals.dtype.kind == 'S':
                        vals = vals.astype(float)
                    raw[feat] = vals.astype(np.float32)
                else:
                    raw[feat] = np.zeros(N, dtype=np.float32)
            standardizer = MetadataStandardizer(fields=METADATA_FEATURES)
            metadata_all = standardizer.transform(raw)  # applies log10, scaling; NaNs → 0
        else:
            metadata_all = None

    # Group positions (0..N-1) by TIC; preserve unique_tics order for output alignment
    unique_tics = np.unique(tic_ids_sorted)
    tic_to_pos  = {t: np.where(tic_ids_sorted == t)[0] for t in unique_tics}

    per_sector_latents   = [None] * N
    cross_sector_latents = []

    model.eval()
    print(f'Extracting latents for {len(unique_tics)} stars '
          f'({N} sectors total, pooling={pooling_mode}, batch_size~{batch_size})...')

    star_buffer   = []
    buffered_secs = 0
    stars_done    = 0

    # Resolve spectra sidecar path (same convention as LazyH5Dataset)
    spectra_h5_path = None
    if use_conv_channels:
        stem = h5_path.rsplit('.h5', 1)[0]
        candidate = stem + '_spectra.h5'
        if Path(candidate).exists():
            spectra_h5_path = candidate
        else:
            with h5py.File(h5_path, 'r') as f_check:
                if 'power' in f_check and 'f_stat' in f_check and 'acf' in f_check:
                    spectra_h5_path = h5_path
        if spectra_h5_path is None:
            raise FileNotFoundError(
                f'Conv channel data not found. Expected sidecar at {candidate} '
                f'or power/f_stat/acf datasets inside {h5_path}.')
        print(f'  Conv channel source: {spectra_h5_path}')

    # Keep H5 file open during the loop — flux/time read per buffer, not upfront
    spectra_ctx = h5py.File(spectra_h5_path, 'r') if (spectra_h5_path and spectra_h5_path != h5_path) else None
    with h5py.File(h5_path, 'r') as f, torch.no_grad():
        sf = spectra_ctx if spectra_ctx is not None else f  # spectra file handle

        def _run_buffer(star_buffer):
            """Forward pass for a list of (pos_array, tic) pairs; fills per_sector_latents
            and appends to cross_sector_latents in the same order as star_buffer."""
            all_pos    = np.concatenate([pos for pos, _ in star_buffer])
            star_sizes = [len(pos) for pos, _ in star_buffer]

            # Map local positions to actual H5 row indices; sort for efficient read
            h5_idx   = sample_indices[all_pos]
            sort_ord = np.argsort(h5_idx)
            inv_ord  = np.argsort(sort_ord)
            h5_idx_s = h5_idx[sort_ord]

            lens_b    = lengths[all_pos]
            batch_max = int(lens_b.max())

            flux_b = torch.tensor(f['flux'][h5_idx_s,     :batch_max][inv_ord], dtype=torch.float32, device=device)
            ferr_b = torch.tensor(f['flux_err'][h5_idx_s, :batch_max][inv_ord], dtype=torch.float32, device=device)
            time_b = torch.tensor(f['time'][h5_idx_s,     :batch_max][inv_ord], dtype=torch.float32, device=device)

            B, L = flux_b.shape
            mask = torch.zeros(B, L, device=device)
            for j, vlen in enumerate(lens_b):
                mask[j, :vlen] = 1.0

            x_in = torch.stack([flux_b, ferr_b], dim=-1)
            t_in = time_b.unsqueeze(-1)

            meta_batch = (torch.tensor(metadata_all[all_pos], dtype=torch.float32, device=device)
                          if metadata_all is not None else None)

            if use_conv_channels and spectra_h5_path is not None:
                conv_data_batch = {
                    'power':  torch.tensor(sf['power'][h5_idx_s][inv_ord],  dtype=torch.float32, device=device),
                    'f_stat': torch.tensor(sf['f_stat'][h5_idx_s][inv_ord], dtype=torch.float32, device=device),
                    'acf':    torch.tensor(sf['acf'][h5_idx_s][inv_ord],    dtype=torch.float32, device=device),
                }
            else:
                conv_data_batch = None

            out   = model(x_in, t_in, mask=mask, metadata=meta_batch,
                          conv_data=conv_data_batch, return_states=True)
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')
            del out, flux_b, ferr_b, x_in, t_in, mask, meta_batch, conv_data_batch

            h_fwd_split = torch.split(h_fwd, star_sizes, dim=0) if h_fwd is not None else [None] * len(star_buffer)
            h_bwd_split = torch.split(h_bwd, star_sizes, dim=0) if h_bwd is not None else [None] * len(star_buffer)
            del h_fwd, h_bwd

            offset = 0
            for (pos, _), hf, hb in zip(star_buffer, h_fwd_split, h_bwd_split):
                h_sectors = [] if compute_cross_sector else None
                t_sectors = [] if compute_cross_sector else None
                for j in range(len(pos)):
                    vlen = int(lens_b[offset + j])
                    if hf is not None and hb is not None:
                        h = torch.cat([hf[j, :vlen, :], hb[j, :vlen, :]], dim=-1)
                    elif hf is not None:
                        h = hf[j, :vlen, :]
                    else:
                        h = hb[j, :vlen, :]
                    t = time_b[offset + j, :vlen]

                    if pooling_mode == 'mean':
                        latent = h.mean(dim=0)
                    elif pooling_mode == 'final':
                        latent = h[-1, :]
                    elif pooling_mode == 'multiscale':
                        latent = compute_multiscale_features(h, t, n_segments=4)
                    else:
                        raise ValueError(f'Unknown pooling_mode: {pooling_mode}')

                    per_sector_latents[pos[j]] = latent.cpu().numpy()
                    if compute_cross_sector:
                        h_sectors.append(h)
                        t_sectors.append(t)
                    del h, t, latent

                if compute_cross_sector:
                    # Cross-sector pooling: stitch all sectors end-to-end on a
                    # synthetic continuous time axis. Each subsequent sector is
                    # offset by the previous sector's t_max plus the median
                    # within-sector cadence, so the inter-sector boundary looks
                    # like one regular cadence step (no discontinuity injected).
                    h_all, t_all = _concat_sectors_with_time(h_sectors, t_sectors)
                    cross_sector_latents.append(compute_multiscale_features(h_all, t_all, n_segments=4).cpu().numpy())
                    del h_sectors, t_sectors, h_all, t_all
                offset += len(pos)

            del h_fwd_split, h_bwd_split, time_b

        for tic in unique_tics:
            pos = tic_to_pos[tic]
            star_buffer.append((pos, tic))
            buffered_secs += len(pos)

            if buffered_secs >= batch_size:
                _run_buffer(star_buffer)
                stars_done   += len(star_buffer)
                star_buffer   = []
                buffered_secs = 0
                if stars_done % 500 < batch_size:
                    print(f'  {stars_done}/{len(unique_tics)} stars processed')

        if star_buffer:
            _run_buffer(star_buffer)

    if spectra_ctx is not None:
        spectra_ctx.close()

    per_sector_arr   = np.stack(per_sector_latents)
    cross_sector_arr = np.stack(cross_sector_latents) if cross_sector_latents else np.empty((0, per_sector_arr.shape[1]))
    print(f'Done: {N} per-sector latents {per_sector_arr.shape}, '
          f'{len(unique_tics)} cross-sector latents {cross_sector_arr.shape}')
    return per_sector_arr, cross_sector_arr, unique_tics


def load_host_ages(h5_path: str, age_csv: str, metadata_csv: str = None,
                   sample_indices: np.ndarray = None):
    """Load host-star ages joined with photometric metadata.

    Mirrors the return signature of plot_umap_latent.load_ages so it can be
    used as a drop-in `age_loader_fn` for load_data_fresh.

    Reads `st_age` (Gyr), `st_ageerr1` (Gyr), `st_ageerr2` (Gyr) from
    `archive_ages_default.csv`-style CSVs and converts to Myr. BPRP0 /
    BPRP0_err / MG_quick / MG_quick_err are joined from `metadata_csv` by
    GaiaDR3_ID. Hosts have no cluster membership prior, so mem_prob is returned
    as NaN (the outlier-mixture loss falls back to the constant 0.9 prior).

    Returns: ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors
    """
    with h5py.File(h5_path, 'r') as f:
        raw_ids = f['metadata']['GaiaDR3_ID'][:]
        tic_ids = f['metadata']['tic'][:]
        sectors = f['metadata']['sector'][:]

    if sample_indices is not None:
        sorted_indices = np.sort(sample_indices)
        raw_ids = raw_ids[sorted_indices]
        tic_ids = tic_ids[sorted_indices]
        sectors = sectors[sorted_indices]

    gaia_ids = np.array([g.decode('utf-8').strip() for g in raw_ids])

    df_age = pd.read_csv(age_csv)
    df_age['GaiaDR3_ID'] = df_age['GaiaDR3_ID'].astype(str)
    if 'st_age' in df_age.columns:
        df_age['age_Myr'] = df_age['st_age'].astype(float) * 1000.0  # Gyr → Myr

    if metadata_csv is not None:
        df_meta = pd.read_csv(metadata_csv)
        df_meta['GaiaDR3_ID'] = df_meta['GaiaDR3_ID'].astype(str)
        # Right-merge so age columns take priority for any overlapping name
        df = df_age.merge(df_meta, on='GaiaDR3_ID', how='left', suffixes=('', '_meta'))
    else:
        df = df_age

    id_to_age       = dict(zip(df['GaiaDR3_ID'], df['age_Myr']))      if 'age_Myr'      in df.columns else {}
    id_to_bprp0     = dict(zip(df['GaiaDR3_ID'], df['BPRP0']))         if 'BPRP0'        in df.columns else {}
    id_to_bprp0_err = dict(zip(df['GaiaDR3_ID'], df['BPRP0_err']))     if 'BPRP0_err'    in df.columns else {}
    id_to_mg        = dict(zip(df['GaiaDR3_ID'], df['MG_quick']))      if 'MG_quick'     in df.columns else {}
    id_to_mg_err    = dict(zip(df['GaiaDR3_ID'], df['MG_quick_err']))  if 'MG_quick_err' in df.columns else {}
    id_to_mem_prob  = dict(zip(df['GaiaDR3_ID'], df['mem_prob_val']))  if 'mem_prob_val' in df.columns else {}

    ages      = np.array([id_to_age.get(g, np.nan)       for g in gaia_ids], dtype=float)
    bprp0     = np.array([id_to_bprp0.get(g, np.nan)     for g in gaia_ids], dtype=float)
    bprp0_err = np.array([id_to_bprp0_err.get(g, np.nan) for g in gaia_ids], dtype=float)
    mg        = np.array([id_to_mg.get(g, np.nan)        for g in gaia_ids], dtype=float)
    mg_err    = np.array([id_to_mg_err.get(g, np.nan)    for g in gaia_ids], dtype=float)
    mem_prob  = np.array([id_to_mem_prob.get(g, np.nan)  for g in gaia_ids], dtype=float)

    n_with_age = np.sum(~np.isnan(ages))
    print(f'Loaded host ages for {n_with_age}/{len(ages)} light curves '
          f'({100*n_with_age/max(len(ages),1):.1f}%)')
    return ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors


def load_data_fresh(model_path: str, h5_path: str, csv_path: str,
                    hidden_size: int, direction: str, mode: str, use_flow: bool,
                    max_length: int = None, batch_size: int = 32, pooling_mode: str = 'multiscale',
                    max_samples: int = None, stratify_by_age: bool = False,
                    n_age_bins: int = 20, bprp0_min: float = None, bprp0_max: float = None,
                    use_metadata: bool = False, use_conv_channels: bool = False,
                    conv_config: dict = None, require_prot: bool = False,
                    device: torch.device = None, pca_h5_paths: list = None,
                    use_mg: bool = False, compute_cross_sector: bool = False,
                    age_loader_fn=None):
    """Load model and extract per-sector and cross-sector latent vectors.

    Returns:
        latent_vectors:          (N, D) per-sector latent vectors
        cross_sector_latents:    (S, D) cross-sector latent vectors (one per unique star)
        cross_sector_tic_ids:    (S,)   TIC ordering for cross_sector_latents
        ages:                    (N,) stellar ages in Myr
        bprp0:                   (N,) BP-RP0 colour
        gaia_ids:                (N,) GaiaDR3_ID strings
        tic_ids:                 (N,) TESS TIC IDs (int64)
        sectors:                 (N,) TESS sector numbers (int64)
    """
    from plot_umap_latent import load_model, load_ages, get_stratified_indices
    from lcgen.models.TimeSeriesDataset import METADATA_FEATURES

    if age_loader_fn is None:
        age_loader_fn = load_ages

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Step 1: load full metadata to build the valid-sample mask ----------
    all_ages, all_bprp0, all_bprp0_err, all_mg, all_mg_err, all_mem_prob, all_gaia_ids, _, _ = age_loader_fn(h5_path, csv_path, sample_indices=None)

    valid_mask = ~np.isnan(all_ages) & ~np.isnan(all_bprp0) & ~np.isnan(all_bprp0_err)
    if use_mg:
        valid_mask &= ~np.isnan(all_mg)
    if bprp0_min is not None:
        valid_mask &= (all_bprp0 >= bprp0_min)
    if bprp0_max is not None:
        valid_mask &= (all_bprp0 <= bprp0_max)

    if require_prot:
        df_csv = pd.read_csv(csv_path)
        df_csv['GaiaDR3_ID'] = df_csv['GaiaDR3_ID'].astype(str)
        id_to_prot = dict(zip(df_csv['GaiaDR3_ID'], df_csv['Prot']))
        all_prot = np.array([id_to_prot.get(g, np.nan) for g in all_gaia_ids])
        valid_mask &= (~np.isnan(all_prot) & (all_prot > 0))
        print(f'Filtering to stars with rotation periods: {valid_mask.sum()} valid samples')

    # -- Step 2: determine sample indices -----------------------------------
    if max_samples is not None:
        if stratify_by_age:
            ages_for_strat = np.where(valid_mask, all_ages, np.nan)
            sample_indices = get_stratified_indices(ages_for_strat, max_samples, n_age_bins)
        else:
            sample_indices = np.where(valid_mask)[0][:max_samples]
    else:
        sample_indices = np.where(valid_mask)[0]

    # Guard against out-of-bounds indices
    with h5py.File(h5_path, 'r') as f:
        h5_size = len(f['flux'])
    ok = sample_indices < h5_size
    if not np.all(ok):
        print(f'Warning: {(~ok).sum()} indices exceed H5 size — dropped.')
        sample_indices = sample_indices[ok]

    # -- Step 3: load metadata for selected samples -------------------------
    # Done before extraction so tic_ids are available for star-grouped processing.
    ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors = age_loader_fn(h5_path, csv_path, sample_indices)

    # Defensive validity filter (load_ages returns sorted order matching sorted sample_indices)
    # mem_prob may be NaN for non-cluster stars — that is valid, do not filter on it
    final_ok        = ~np.isnan(ages) & ~np.isnan(bprp0) & ~np.isnan(bprp0_err)
    if use_mg:
        final_ok &= ~np.isnan(mg)
    sorted_indices  = np.sort(sample_indices)[final_ok]
    ages      = ages[final_ok]
    bprp0     = bprp0[final_ok]
    bprp0_err = bprp0_err[final_ok]
    mg        = mg[final_ok]
    mg_err    = mg_err[final_ok]
    mem_prob  = mem_prob[final_ok]
    gaia_ids  = gaia_ids[final_ok]
    tic_ids   = tic_ids[final_ok]
    sectors   = sectors[final_ok]

    # -- Step 4: load model -------------------------------------------------
    num_meta_features = len(METADATA_FEATURES) if use_metadata else 0
    model = load_model(model_path, device,
                       hidden_size=hidden_size, direction=direction, mode=mode,
                       use_flow=use_flow, num_meta_features=num_meta_features,
                       use_conv_channels=use_conv_channels, conv_config=conv_config)

    actual_use_metadata = model.num_meta_features > 0
    actual_use_conv     = model.use_conv_channels

    # -- Step 5: extract latents for ALL stars in h5_path (one pass) --------
    # We extract everything at once and then filter to the labeled subset.
    # This avoids re-processing h5_path in the PCA step.
    with h5py.File(h5_path, 'r') as f:
        all_tic_ids_h5 = f['metadata']['tic'][:]
    all_h5_indices = np.arange(len(all_tic_ids_h5))

    print('\n=== Extracting latents for all stars in h5_path ===')
    all_per_sector, all_cross_sector, all_cross_tic_ids = extract_latents_by_star(
        model, h5_path, device,
        sample_indices=all_h5_indices,
        tic_ids_sorted=all_tic_ids_h5,
        max_length=max_length,
        pooling_mode=pooling_mode,
        batch_size=batch_size,
        use_metadata=actual_use_metadata,
        use_conv_channels=actual_use_conv,
        compute_cross_sector=compute_cross_sector,
    )

    # Filter per-sector latents to the labeled subset (sorted_indices are H5 row indices)
    latent_vectors = all_per_sector[sorted_indices]

    # Filter cross-sector latents to labeled TIC IDs (only populated if compute_cross_sector=True)
    if compute_cross_sector and len(all_cross_sector) > 0:
        labeled_tic_set = set(tic_ids.tolist())
        labeled_cross_mask = np.isin(all_cross_tic_ids, list(labeled_tic_set))
        cross_sector_latents = all_cross_sector[labeled_cross_mask]
        cross_sector_tic_ids = all_cross_tic_ids[labeled_cross_mask]
    else:
        cross_sector_latents = None
        cross_sector_tic_ids = None

    print(f'Loaded {len(ages)} LC samples | latent dim: {latent_vectors.shape[1]}')
    print(f'Age range: {ages.min():.1f} – {ages.max():.1f} Myr')
    print(f'Unique stars: {len(np.unique(tic_ids))}')

    # -- Step 6: build PCA latent set (h5_path already extracted; add extra files) --
    pca_parts = [all_per_sector]
    if pca_h5_paths:
        from plot_umap_latent import extract_latent_vectors as _extract_all
        extra_paths = [p for p in pca_h5_paths if Path(p).resolve() != Path(h5_path).resolve()]
        if extra_paths:
            print('\n=== Extracting all-star latents from additional PCA H5 files ===')
            for p in extra_paths:
                print(f'  {p}')
                lv = _extract_all(model, p, device,
                                  max_length=max_length, batch_size=batch_size,
                                  pooling_mode=pooling_mode, sample_indices=None,
                                  use_metadata=actual_use_metadata,
                                  use_conv_channels=actual_use_conv)
                pca_parts.append(lv)
    pca_all_latents = np.concatenate(pca_parts, axis=0)
    print(f'Total PCA latents: {len(pca_all_latents)} (from {1 + len(extra_paths if pca_h5_paths else [])} file(s))')

    return latent_vectors, cross_sector_latents, cross_sector_tic_ids, ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors, pca_all_latents


# ---------------------------------------------------------------------------
# Latents cache  (save once, load many times)
# ---------------------------------------------------------------------------

def save_latents_cache(cache_path: str,
                       latent_vectors, ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors,
                       cross_sector_latents=None, cross_sector_tic_ids=None,
                       mg=None, mg_err=None, mem_prob=None):
    """Save per-sector latents + identifiers (and optionally cross-sector latents) to npz.

    The cache contains everything needed for all star_aggregation modes so that
    the model never needs to be re-run when comparing methods.

    Layout
    ------
    latent_vectors        (N, D)          per-sector pooled latents
    ages                  (N,)            stellar age in Myr
    bprp0                 (N,)            BP-RP0
    bprp0_err             (N,)            BP-RP0 uncertainty
    gaia_ids              (N,)  str       GaiaDR3_ID
    tic_ids               (N,)  int64     TESS TIC
    sectors               (N,)  int64     TESS sector number
    mg                    (N,)            absolute G mag         [optional]
    mem_prob              (N,)            cluster membership prob [optional]
    cross_sector_latents  (S, D)          one multiscale latent per unique star  [optional]
    cross_sector_tic_ids  (S,)  int64     TIC ordering for cross_sector_latents  [optional]
    """
    save_dict = dict(
        latent_vectors=latent_vectors,
        ages=ages, bprp0=bprp0, bprp0_err=bprp0_err,
        gaia_ids=gaia_ids.astype(str),
        tic_ids=tic_ids, sectors=sectors,
    )
    if mg is not None:
        save_dict['mg'] = mg
    if mg_err is not None:
        save_dict['mg_err'] = mg_err
    if mem_prob is not None:
        save_dict['mem_prob'] = mem_prob
    if cross_sector_latents is not None:
        save_dict['cross_sector_latents'] = cross_sector_latents
        save_dict['cross_sector_tic_ids'] = cross_sector_tic_ids
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save_dict)
    keys = list(save_dict.keys())
    print(f'Saved latents cache → {cache_path}  (keys: {keys})')


def load_latents_cache(cache_path: str):
    """Load a latents cache written by save_latents_cache.

    Returns
    -------
    dict with keys: latent_vectors, ages, bprp0, bprp0_err, mg, mg_err, mem_prob,
    gaia_ids, tic_ids, sectors, cross_sector_latents, cross_sector_tic_ids.
    Optional fields are None when the cache was written without them
    (legacy caches: bprp0_err / mg / mg_err / mem_prob may need a CSV-based fill).
    """
    data = np.load(cache_path, allow_pickle=True)
    out = {
        'latent_vectors': data['latent_vectors'],
        'ages':           data['ages'],
        'bprp0':          data['bprp0'],
        'gaia_ids':       data['gaia_ids'],
        'tic_ids':        data['tic_ids'],
        'sectors':        data['sectors'],
        'bprp0_err':      data['bprp0_err'] if 'bprp0_err' in data else None,
        'mg':             data['mg']        if 'mg'        in data else None,
        'mg_err':         data['mg_err']    if 'mg_err'    in data else None,
        'mem_prob':       data['mem_prob']  if 'mem_prob'  in data else None,
        'cross_sector_latents': data['cross_sector_latents'] if 'cross_sector_latents' in data else None,
        'cross_sector_tic_ids': data['cross_sector_tic_ids'] if 'cross_sector_tic_ids' in data else None,
    }
    print(f'Loaded latents cache ← {cache_path}')
    print(f'  per-sector samples : {len(out["ages"])}')
    print(f'  latent dim         : {out["latent_vectors"].shape[1]}')
    if out['cross_sector_latents'] is not None:
        print(f'  cross-sector stars : {len(out["cross_sector_tic_ids"])}')
    else:
        print(f'  cross-sector latents: not in cache')
    missing = [k for k in ('bprp0_err', 'mg', 'mg_err', 'mem_prob') if out[k] is None]
    if missing:
        print(f'  legacy cache (missing: {missing}) — will fill from CSV')
    return out


# ---------------------------------------------------------------------------
# Global PCA artifact (compute once, reuse across sweep + LOSO runs)
# ---------------------------------------------------------------------------

def fit_global_pca_bundle(pca_latents, max_dim, whiten=True):
    """Fit X-normalization stats + PCA at `max_dim` components on the global pool.

    Returns (pca_obj, X_mean, X_std). The PCA is fit on z-scored latents — every
    fold that uses this bundle just truncates pca_obj.components_ to its desired
    dim and projects (after applying the same X normalization). Whiten matches
    the per-fold behavior so downstream model code is unchanged.
    """
    pca_latents = np.asarray(pca_latents, dtype=np.float32)
    X_mean = pca_latents.mean(axis=0).astype(np.float64)
    X_std  = pca_latents.std(axis=0).astype(np.float64) + 1e-8
    X_norm = ((pca_latents - X_mean) / X_std).astype(np.float32)
    max_dim = int(min(max_dim, X_norm.shape[1], X_norm.shape[0]))
    pca = PCA(n_components=max_dim, whiten=whiten)
    pca.fit(X_norm)
    var_expl = pca.explained_variance_ratio_.sum()
    print(f'  Global PCA: fit {max_dim} components on {len(X_norm)} latents '
          f'({100*var_expl:.1f}% cum. variance explained)')
    return pca, X_mean, X_std


def save_global_pca_artifact(path, pca, X_mean, X_std):
    """Persist the bundle so subsequent runs reuse the exact same basis."""
    np.savez(path,
        pca_mean=pca.mean_,                        # (D,)  feature-wise mean used by PCA.transform
        pca_components=pca.components_,            # (K, D)
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        pca_singular_values=pca.singular_values_,
        pca_n_components=np.int64(pca.n_components_),
        pca_n_features_in=np.int64(pca.n_features_in_),
        pca_n_samples=np.int64(pca.n_samples_),
        pca_noise_variance=np.float64(getattr(pca, 'noise_variance_', 0.0)),
        pca_whiten=np.bool_(pca.whiten),
        X_mean=X_mean, X_std=X_std)
    print(f'  Saved global PCA artifact → {path}')


def load_global_pca_artifact(path, want_dim):
    """Reconstruct an sklearn PCA truncated to the first `want_dim` components."""
    d = np.load(path, allow_pickle=False)
    n_cached = int(d['pca_n_components'])
    if want_dim > n_cached:
        raise ValueError(
            f'load_global_pca_artifact({path}): requested {want_dim} components '
            f'but cache has only {n_cached}. Recompute with --pca_cache_max_dim '
            f'>= {want_dim}.')
    pca = PCA(n_components=want_dim, whiten=bool(d['pca_whiten']))
    pca.mean_                    = d['pca_mean']
    pca.components_              = d['pca_components'][:want_dim]
    pca.explained_variance_      = d['pca_explained_variance'][:want_dim]
    pca.explained_variance_ratio_= d['pca_explained_variance_ratio'][:want_dim]
    pca.singular_values_         = d['pca_singular_values'][:want_dim]
    pca.n_components_            = want_dim
    pca.n_features_in_           = int(d['pca_n_features_in'])
    pca.n_samples_               = int(d['pca_n_samples'])
    pca.noise_variance_          = float(d['pca_noise_variance'])
    X_mean = d['X_mean'].astype(np.float64)
    X_std  = d['X_std'].astype(np.float64)
    var_expl = pca.explained_variance_ratio_.sum()
    print(f'  Loaded global PCA artifact ← {path} '
          f'(took {want_dim}/{n_cached} components, {100*var_expl:.1f}% var explained)')
    return pca, X_mean, X_std


# ---------------------------------------------------------------------------
# Star-level aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_star(latent_vectors, ages, bprp0, bprp0_err, mg, mg_err, mem_prob, tic_ids, gaia_ids,
                      method: str, source=None):
    """Reduce per-sector arrays to one row per unique star.

    Groups by GaiaID (not TIC ID) as the primary key, since occasional cases
    exist where multiple TIC IDs map to the same Gaia source.

    Args:
        method: one of 'latent_mean', 'latent_median', 'latent_max',
                'latent_mean_std'.
                'latent_mean_std' concatenates [mean, std] along the feature
                axis, doubling the latent dimension. For single-sector stars
                the std is all zeros — the MLP sees this as a valid input but
                it carries no sector-variability information. Be aware this may
                introduce a confound if the number of sectors observed correlates
                with stellar properties.

    Returns:
        star_latents:    (n_stars, D)  — D is 2× input dim for latent_mean_std
        star_ages:       (n_stars,)
        star_bprp0:      (n_stars,)
        star_bprp0_err:  (n_stars,)
        star_mg:         (n_stars,)
        star_mg_err:     (n_stars,)
        star_mem_prob:   (n_stars,)
        unique_gaia_ids: (n_stars,)
    """
    unique_gids    = np.unique(gaia_ids)
    star_latents   = []
    star_ages      = []
    star_bprp0     = []
    star_bprp0_err = []
    star_mg        = []
    star_mg_err    = []
    star_mem_prob  = []
    star_source    = [] if source is not None else None

    for gid in unique_gids:
        mask = gaia_ids == gid
        sectors_latents = latent_vectors[mask]  # (n_sectors, D)
        if method == 'latent_mean':
            star_latents.append(sectors_latents.mean(axis=0))
        elif method == 'latent_median':
            star_latents.append(np.median(sectors_latents, axis=0))
        elif method == 'latent_max':
            star_latents.append(sectors_latents.max(axis=0))
        elif method == 'latent_mean_std':
            mean = sectors_latents.mean(axis=0)
            std  = sectors_latents.std(axis=0)   # zero for single-sector stars
            star_latents.append(np.concatenate([mean, std]))
        else:
            raise ValueError(f'Unknown aggregation method: {method}')
        star_ages.append(ages[mask][0])
        star_bprp0.append(bprp0[mask][0])
        star_bprp0_err.append(bprp0_err[mask][0])
        star_mg.append(mg[mask][0])
        star_mg_err.append(mg_err[mask][0])
        star_mem_prob.append(mem_prob[mask][0])
        if star_source is not None:
            star_source.append(source[mask][0])

    star_latents = np.stack(star_latents)
    print(f'{method}: reduced to {len(unique_gids)} stars | latent dim: {star_latents.shape[1]}')
    out = (star_latents, np.array(star_ages), np.array(star_bprp0), np.array(star_bprp0_err),
           np.array(star_mg), np.array(star_mg_err), np.array(star_mem_prob), unique_gids)
    if star_source is not None:
        out = out + (np.array(star_source, dtype=int),)
    return out


def extract_cross_sector_latents(model, h5_path: str, device: torch.device,
                                  sample_indices: np.ndarray, tic_ids_sorted: np.ndarray,
                                  max_length: int = 16384):
    """Pool hidden states across ALL sectors of each star via multiscale aggregation.

    Concatenates the valid hidden states from every sector of a star into a single
    sequence, then applies the same multiscale pooling used per-sector. This lets
    the model see the full temporal context of the star before collapsing to a
    single latent vector.

    Args:
        model:           trained BiDirectionalMinGRU
        h5_path:         path to H5 data file
        device:          torch device
        sample_indices:  H5 indices in sorted order (same ordering load_data_fresh uses)
        tic_ids_sorted:  TIC IDs aligned to sample_indices (same sorted order)
        max_length:      per-sector max sequence length

    Returns:
        star_latents:    (n_unique_stars, D)
        unique_tic_ids:  (n_unique_stars,)
    """
    from plot_umap_latent import compute_multiscale_features

    sorted_idx = np.sort(sample_indices)

    with h5py.File(h5_path, 'r') as f:
        flux_all     = f['flux'][sorted_idx]
        flux_err_all = f['flux_err'][sorted_idx]
        time_all     = f['time'][sorted_idx]
        lengths      = f['length'][sorted_idx]

    # Group positions in sorted arrays by TIC
    unique_tics      = np.unique(tic_ids_sorted)
    tic_to_positions = {t: np.where(tic_ids_sorted == t)[0] for t in unique_tics}

    star_latents = []
    model.eval()

    print(f'Cross-sector extraction for {len(unique_tics)} stars...')

    with torch.no_grad():
        for i, tic in enumerate(unique_tics):
            pos = tic_to_positions[tic]  # indices into sorted arrays

            flux_b = torch.tensor(flux_all[pos],     dtype=torch.float32, device=device)
            ferr_b = torch.tensor(flux_err_all[pos], dtype=torch.float32, device=device)
            time_b = torch.tensor(time_all[pos],     dtype=torch.float32, device=device)
            lens_b = lengths[pos]

            actual_max = min(max_length, flux_b.shape[1])
            flux_b = flux_b[:, :actual_max]
            ferr_b = ferr_b[:, :actual_max]
            time_b = time_b[:, :actual_max]
            lens_b = np.minimum(lens_b, actual_max)

            B, L = flux_b.shape
            mask = torch.zeros(B, L, device=device)
            for j, vlen in enumerate(lens_b):
                mask[j, :vlen] = 1.0

            x_in = torch.stack([flux_b, ferr_b], dim=-1)
            t_in = time_b.unsqueeze(-1)

            out   = model(x_in, t_in, mask=mask, return_states=True)
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')

            # Collect valid hidden states from every sector of this star
            h_segments = []
            t_segments = []
            for j in range(B):
                vlen = lens_b[j]
                if h_fwd is not None and h_bwd is not None:
                    h = torch.cat([h_fwd[j, :vlen, :], h_bwd[j, :vlen, :]], dim=-1)
                elif h_fwd is not None:
                    h = h_fwd[j, :vlen, :]
                else:
                    h = h_bwd[j, :vlen, :]
                h_segments.append(h)
                t_segments.append(time_b[j, :vlen])

            h_all, t_all = _concat_sectors_with_time(h_segments, t_segments)
            latent = compute_multiscale_features(h_all, t_all, n_segments=4)
            star_latents.append(latent.cpu().numpy())

            if (i + 1) % 250 == 0:
                print(f'  {i + 1}/{len(unique_tics)} stars processed')

    star_latents = np.stack(star_latents)
    print(f'Cross-sector latents: {len(unique_tics)} stars | shape {star_latents.shape}')
    return star_latents, unique_tics


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

class CombinedLRScheduler:
    """Exponential decay multiplied by cosine annealing (mirrors ChronoFlow notebook).

    lr = initial_lr * decay_rate^epoch * 0.5 * (1 + cos(π * epoch / T_max))

    decay_rate controls how aggressively the lr falls over training.
    A value of 10**(-3 / n_epochs) reduces the base lr by 1000x over the full run.
    T_max is the cosine period; setting it to n_epochs gives one full cosine cycle.
    """
    def __init__(self, optimizer, n_epochs: int, initial_lr: float,
                 decay_rate: float = None):
        self.optimizer    = optimizer
        self.T_max        = n_epochs
        self.initial_lr   = initial_lr
        self.decay_rate   = decay_rate if decay_rate is not None else 10 ** (-3.0 / n_epochs)
        self.current_epoch = 0
        # Capture each group's current LR as its base. Schedule scales all
        # groups by the same factor so per-group ratios (e.g. encoder LR being
        # 0.001× the flow LR) are preserved across steps.
        for pg in optimizer.param_groups:
            pg.setdefault('base_lr', pg['lr'])

    def step(self):
        exp_factor    = self.decay_rate ** self.current_epoch
        cos_factor    = 0.5 * (1.0 + np.cos(np.pi * self.current_epoch / self.T_max))
        scale         = exp_factor * cos_factor
        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['base_lr'] * scale
        self.current_epoch += 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _batched_predict_stats(model, Z_val_t, bprp0_val_t, berr_val_t, mg_val_t,
                            loga_grid, pred_batch_size=256):
    """Run predict_stats in chunks to avoid the (B × G) memory spike.

    With a 1000-point age grid, calling predict_stats on the full val set at once
    creates tensors of shape (B*1000, D) which can be hundreds of MB. Chunking to
    pred_batch_size stars keeps peak memory proportional to the chunk size.
    """
    stat_keys = ('median', 'mean', 'map', 'p16', 'p84')
    accum = {k: [] for k in stat_keys}
    n = Z_val_t.shape[0]
    for start in range(0, n, pred_batch_size):
        end = min(start + pred_batch_size, n)
        stats = model.predict_stats(
            Z_val_t[start:end], bprp0_val_t[start:end],
            berr_val_t[start:end], mg_val_t[start:end], loga_grid
        )
        for k in stat_keys:
            accum[k].append(stats[k].cpu())
    return {k: torch.cat(accum[k]) for k in stat_keys}


def train_single_fold(X_train, y_train, bprp0_train, bprp0_err_train, mg_train, mem_prob_train,
                      X_val, y_val, bprp0_val, bprp0_err_val, mg_val, mem_prob_val,
                      pca_dim, lr, weight_decay, n_epochs, batch_size, device,
                      flow_transforms=8, flow_hidden_features=None, loga_grid=None,
                      encoder_type='pca', mlp_encoder_hidden=None, aux_loss_weight=1.0,
                      use_mg=False, lr_decay_rate=None, global_pca=None,
                      dropout=0.1, variance_reg_weight=0.0,
                      training_stages='joint', encoder_pretrain_epochs=100,
                      joint_finetune_epochs=50, finetune_encoder_lr_mult=0.01,
                      finetune_flow_lr_mult=0.1,
                      prediction_mode='nle',
                      sectors_idx_train=None, n_sectors=0,
                      balance_sector_age=False, n_balance_age_bins=10,
                      adv_sector_weight=0.0, adv_hidden=64):
    """Train one fold of NLE *or* NPE model.

    prediction_mode:
        'nle' — flow models p(z | log_age, colours), z is the bottleneck. Default.
        'npe' — flow models p(log_age | z, colours) directly. Requires an MLP/linear
                encoder (PCA-only mode is not supported for NPE).

    encoder_type='pca': PCA fit on training fold only (no leakage, no collapse).
        If global_pca is provided, it is used as-is (pre-fit on all stars) and
        only transform() is called — no per-fold fitting.
    encoder_type='mlp' or 'linear': Learned encoder with auxiliary age L1 loss.
        No PCA step; raw z-scored latents used.

    training_stages:
        'joint'       — train all params together (current default)
        'two_stage'   — stage 1: encoder+aux only; stage 2: flow only
        'three_stage' — stage 1: encoder+aux; stage 2: flow; stage 3: joint fine-tune

    Returns:
        val_stats, best_loss, best_state, fold_pca, train_losses, val_losses
        (fold_pca is None when encoder_type='mlp' or 'linear')
    """
    if loga_grid is None:
        loga_grid = torch.tensor(LOGA_GRID, dtype=torch.float32, device=device)

    fold_pca = None

    # Sector-aware training switches (both need the NLE MLP/linear encoder).
    use_adv     = adv_sector_weight > 0 and n_sectors > 0 and sectors_idx_train is not None
    use_balance = balance_sector_age and sectors_idx_train is not None
    if use_adv and (encoder_type == 'pca' or prediction_mode != 'nle'):
        raise ValueError("adv_sector_weight requires encoder_type in {mlp, linear} and "
                         "prediction_mode='nle' (the adversary attaches to the MLP bottleneck).")

    if prediction_mode == 'npe' and encoder_type == 'pca':
        raise ValueError("prediction_mode='npe' requires encoder_type='mlp' or 'linear' "
                         "(NPE conditions on a learned bottleneck — PCA-only NPE is "
                         "not supported).")

    if encoder_type == 'pca':
        if global_pca is not None:
            fold_pca = global_pca
            Z_train  = fold_pca.transform(X_train).astype(np.float32)
            Z_val    = fold_pca.transform(X_val).astype(np.float32)
        else:
            fold_pca = PCA(n_components=pca_dim, whiten=True)
            Z_train  = fold_pca.fit_transform(X_train).astype(np.float32)
            Z_val    = fold_pca.transform(X_val).astype(np.float32)
        var_expl  = fold_pca.explained_variance_ratio_.sum()
        print(f'    PCA {pca_dim}D: {100*var_expl:.1f}% variance explained')
        model = AgePredictor(pca_dim, flow_transforms=flow_transforms,
                             flow_hidden_features=flow_hidden_features,
                             use_mg=use_mg).to(device)
    else:  # mlp or linear
        Z_train = X_train.astype(np.float32)
        Z_val   = X_val.astype(np.float32)
        input_dim = Z_train.shape[1]
        if encoder_type == 'linear':
            mlp_hidden = []
            print(f'    Linear encoder {input_dim}D → {pca_dim}D bottleneck '
                  f'({prediction_mode.upper()})')
        else:
            mlp_hidden = mlp_encoder_hidden
            print(f'    MLP encoder {input_dim}D → {pca_dim}D bottleneck '
                  f'({prediction_mode.upper()})')
        if prediction_mode == 'npe':
            grid_range = (float(loga_grid.min().item()), float(loga_grid.max().item()))
            model = AgePredictorNPE(
                input_dim=input_dim,
                bottleneck_dim=pca_dim,
                mlp_hidden=mlp_hidden,
                flow_transforms=flow_transforms,
                flow_hidden_features=flow_hidden_features,
                aux_loss_weight=aux_loss_weight,
                use_mg=use_mg,
                dropout=dropout,
                variance_reg_weight=variance_reg_weight,
                age_grid_range=grid_range,
            ).to(device)
        else:
            model = AgePredictorMLP(
                input_dim=input_dim,
                bottleneck_dim=pca_dim,
                mlp_hidden=mlp_hidden,
                flow_transforms=flow_transforms,
                flow_hidden_features=flow_hidden_features,
                aux_loss_weight=aux_loss_weight,
                use_mg=use_mg,
                dropout=dropout,
                variance_reg_weight=variance_reg_weight,
                n_sectors=(n_sectors if use_adv else 0),
                adv_hidden=adv_hidden,
            ).to(device)

    # Sector index per row (zeros placeholder when sector-aware training is off);
    # always present so the batch loop can unpack a fixed-width tuple.
    sec_train_np = (sectors_idx_train.astype(np.int64) if sectors_idx_train is not None
                    else np.zeros(len(Z_train), dtype=np.int64))
    train_dataset = TensorDataset(
        torch.tensor(Z_train,         dtype=torch.float32, device=device),
        torch.tensor(y_train,         dtype=torch.float32, device=device),
        torch.tensor(bprp0_train,     dtype=torch.float32, device=device),
        torch.tensor(bprp0_err_train, dtype=torch.float32, device=device),
        torch.tensor(mg_train,        dtype=torch.float32, device=device),
        torch.tensor(mem_prob_train,  dtype=torch.float32, device=device),
        torch.tensor(sec_train_np,    dtype=torch.long,    device=device),
    )
    if use_balance:
        # Flatten the (sector × age-bin) joint so over-represented cluster cells
        # stop dominating the gradient and the model can't lean on sector→age.
        from torch.utils.data import WeightedRandomSampler
        y_bin_src = y_train if y_train.ndim == 1 else y_train.mean(axis=1)
        edges   = np.linspace(np.min(y_bin_src), np.max(y_bin_src) + 1e-6, n_balance_age_bins + 1)
        age_bin = np.clip(np.digitize(y_bin_src, edges) - 1, 0, n_balance_age_bins - 1)
        cell    = sectors_idx_train.astype(np.int64) * n_balance_age_bins + age_bin
        uniq_cell, counts = np.unique(cell, return_counts=True)
        cell_count = dict(zip(uniq_cell.tolist(), counts.tolist()))
        w = np.array([1.0 / cell_count[int(c)] for c in cell], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(w / w.sum(), dtype=torch.double),
                                        num_samples=len(Z_train), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        print(f'    balance_sector_age: flattening {len(uniq_cell)} (sector × age-bin) '
              f'cells over {len(Z_train)} rows.')
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Z_val_t        = torch.tensor(Z_val,         dtype=torch.float32, device=device)
    bprp0_val_t    = torch.tensor(bprp0_val,     dtype=torch.float32, device=device)
    berr_val_t     = torch.tensor(bprp0_err_val, dtype=torch.float32, device=device)
    mg_val_t       = torch.tensor(mg_val,        dtype=torch.float32, device=device)
    mem_prob_val_t = torch.tensor(mem_prob_val,  dtype=torch.float32, device=device)
    y_val_t        = torch.tensor(y_val,         dtype=torch.float32, device=device)

    train_losses = []
    val_losses   = []
    best_loss  = float('inf')
    best_state = None
    epochs_done = 0            # global epoch counter for the adversary λ ramp
    adv_total   = max(n_epochs, 1)

    def _run_stage(params, n_stage_epochs, stage_lr, loss_mode='full', label='',
                   optimizer=None):
        """Run one training stage, updating train_losses/val_losses/best in-place.

        If `optimizer` is provided, reuse it (preserves AdamW momentum/variance
        state across stages). Otherwise build a fresh AdamW from `params`.
        Returns the optimizer used so the caller can pass it to the next stage.
        """
        nonlocal best_loss, best_state, epochs_done
        opt = optimizer if optimizer is not None else optim.AdamW(
            params, lr=stage_lr, weight_decay=weight_decay)
        sched = CombinedLRScheduler(opt, n_epochs=n_stage_epochs, initial_lr=stage_lr,
                                    decay_rate=lr_decay_rate)
        for _ in range(n_stage_epochs):
            # DANN λ ramp 0→adv_sector_weight over the full epoch budget — a high λ
            # from step 0 collapses the bottleneck before it learns anything.
            if use_adv:
                p = min(epochs_done / max(adv_total - 1, 1), 1.0)
                cur_lambda = adv_sector_weight * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            else:
                cur_lambda = 0.0
            model.train()
            epoch_loss = 0.0
            for Z_b, y_b, bprp0_b, bprp0_err_b, mg_b, mem_prob_b, sec_b in train_loader:
                opt.zero_grad()
                if isinstance(model, AgePredictorMLP):
                    loss = model(Z_b, y_b, bprp0_b, bprp0_err_b, mg_b, mem_prob_b,
                                 loss_mode=loss_mode,
                                 sector_idx=(sec_b if use_adv else None),
                                 adv_lambda=cur_lambda)
                elif isinstance(model, AgePredictorNPE):
                    loss = model(Z_b, y_b, bprp0_b, bprp0_err_b, mg_b, mem_prob_b,
                                 loss_mode=loss_mode)
                else:
                    loss = model(Z_b, y_b, bprp0_b, bprp0_err_b, mg_b, mem_prob_b)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(Z_b)
            epoch_loss /= len(Z_train)
            train_losses.append(epoch_loss)
            epochs_done += 1

            model.eval()
            with torch.no_grad():
                # Always use NLL-only for model selection — the combined loss
                # (NLL + scaled aux) can cancel to ~0 and is not a useful metric
                if isinstance(model, (AgePredictorMLP, AgePredictorNPE)):
                    val_nll = model(Z_val_t, y_val_t, bprp0_val_t, berr_val_t,
                                    mg_val_t, mem_prob_val_t, loss_mode='nll_only').item()
                else:
                    val_nll = model(Z_val_t, y_val_t, bprp0_val_t, berr_val_t,
                                    mg_val_t, mem_prob_val_t).item()
            val_losses.append(val_nll)
            model.train()

            if val_nll < best_loss:
                best_loss  = val_nll
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            sched.step()
        if label:
            print(f'      {label}: {n_stage_epochs} epochs, best_val_nll={best_loss:.4f}')
        return opt

    use_staged = (training_stages in ('two_stage', 'three_stage')
                  and isinstance(model, (AgePredictorMLP, AgePredictorNPE)))

    if use_staged:
        # Stage 1: encoder + aux_head only (flow frozen)
        for p in model.flow.parameters():
            p.requires_grad_(False)
        encoder_params = list(model.encoder.parameters()) + list(model.aux_age_head.parameters())
        _run_stage(encoder_params, encoder_pretrain_epochs, lr,
                   loss_mode='aux_only', label='Stage 1 (encoder)')

        # Stage 2: flow only (encoder + aux_head frozen)
        # Reset best_loss — stage 1 val used full loss with untrained flow (meaningless)
        best_loss = float('inf')
        best_state = None
        for p in model.flow.parameters():
            p.requires_grad_(True)
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        for p in model.aux_age_head.parameters():
            p.requires_grad_(False)
        flow_epochs = n_epochs - encoder_pretrain_epochs
        if training_stages == 'three_stage':
            flow_epochs = n_epochs - encoder_pretrain_epochs - joint_finetune_epochs
        flow_epochs = max(flow_epochs, 1)
        stage2_opt = _run_stage(list(model.flow.parameters()), flow_epochs, lr,
                                loss_mode='nll_only', label='Stage 2 (flow)')

        # Stage 3 (three_stage only): joint fine-tuning. Reuses stage 2's
        # optimizer (preserves AdamW momentum/variance for the flow) and adds
        # the encoder + aux_head as a second param group. Flow LR is bumped to
        # `lr * finetune_flow_lr_mult` (stage 2's cosine had decayed it to ~0)
        # so the flow has headroom to keep adapting alongside the encoder.
        if training_stages == 'three_stage' and joint_finetune_epochs > 0:
            for p in model.parameters():
                p.requires_grad_(True)
            flow_lr_s3    = lr * finetune_flow_lr_mult
            encoder_lr_s3 = lr * finetune_encoder_lr_mult
            # Reset flow group LR + base_lr; the new scheduler will scale from base_lr.
            for pg in stage2_opt.param_groups:
                pg['lr']      = flow_lr_s3
                pg['base_lr'] = flow_lr_s3
            stage2_opt.add_param_group({
                'params': list(model.encoder.parameters()) + list(model.aux_age_head.parameters()),
                'lr':      encoder_lr_s3,
                'base_lr': encoder_lr_s3,
            })
            print(f'      Stage 3 setup: flow_lr={flow_lr_s3:.2e}, '
                  f'encoder_lr={encoder_lr_s3:.2e} (reusing stage 2 optimizer state)')
            _run_stage(None, joint_finetune_epochs, lr, loss_mode='full',
                       label='Stage 3 (joint)', optimizer=stage2_opt)

        # Ensure all params are unfrozen for prediction
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        # Joint training (original behavior)
        _run_stage(list(model.parameters()), n_epochs, lr, loss_mode='full')

    if best_state is None:
        print('      Warning: val NLL never improved from inf (possible NaN loss). '
              'Using final model weights as fallback.')
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        stats     = _batched_predict_stats(model, Z_val_t, bprp0_val_t, berr_val_t, mg_val_t,
                                           loga_grid.to(device))
        val_stats = {k: v.numpy() for k, v in stats.items()}

    return val_stats, best_loss, best_state, fold_pca, train_losses, val_losses


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def loocv_age_folds(ages, young_n=20, young_size=5, indiv_min=100, sparse_split=1000.0):
    """Leave-one-cluster-out folds for ChronoFlow, using age as a cluster proxy.

    Each ChronoFlow cluster carries one isochrone age, so unique ages ≈ clusters.
    Sort unique ages ascending, then:
      - bundle the youngest `young_n` ages into folds of `young_size` (the young end
        is densely clumped within age uncertainty — testing them individually is
        meaningless and leaves tiny folds);
      - among the rest, keep ages with >= `indiv_min` stars as their own fold
        (well-populated, well-separated clusters — the meaningful LOCO tests);
      - bundle the remaining sparse ages into two folds split at `sparse_split` Myr
        (intermediate vs old field-like).
    SHARED by kfold_age_inference and kfold_gyro_baseline so the folds are identical.
    Returns (fold_of_sample int array aligned to `ages`, n_folds).
    """
    key = np.round(np.asarray(ages, dtype=np.float64), 3)
    uniq, counts = np.unique(key, return_counts=True)            # ascending
    cnt = dict(zip(uniq.tolist(), counts.tolist()))
    age_to_fold, fid = {}, 0
    n_young = min(young_n, len(uniq))
    for s in range(0, n_young, young_size):                       # young bundles
        for a in uniq[s:s + young_size]:
            age_to_fold[a] = fid
        fid += 1
    rest = uniq[n_young:]
    indiv      = [a for a in rest if cnt[a] >= indiv_min]
    sparse_mid = [a for a in rest if cnt[a] < indiv_min and a <  sparse_split]
    sparse_old = [a for a in rest if cnt[a] < indiv_min and a >= sparse_split]
    for a in indiv:                                               # individual clusters
        age_to_fold[a] = fid; fid += 1
    if sparse_mid:
        for a in sparse_mid:
            age_to_fold[a] = fid
        fid += 1
    if sparse_old:
        for a in sparse_old:
            age_to_fold[a] = fid
        fid += 1
    fold_of_sample = np.array([age_to_fold[a] for a in key], dtype=int)
    return fold_of_sample, fid


def loso_sector_folds(star_sectors, n_folds=10, seed=42):
    """Leave-one-sector-out folds for per-star validation (balanced).

    Partitions the unique TESS sectors into `n_folds` disjoint groups. Each held-out
    group defines one fold. For that fold:
      - VALIDATION = stars assigned to the fold (each star is assigned to ONE fold,
        a group among those its own sectors touch — so every validated star genuinely
        has a light curve in a held-out sector);
      - TRAINING eligibility = a star may train ONLY if NONE of its sectors fall in
        the held-out group. A star observed in any held-out sector is removed from
        training entirely (the whole star, not just its held-out-sector rows) — so
        the age model never sees a star that shares a sector with the validation set.

    This is the per-star (latent_max) analogue of the per-sector sector-disjoint CV:
    it estimates generalization to stars observed only in sectors the age model has
    never trained on. A star whose sectors span several groups is excluded from
    training in each of those folds but validated in only one (used nowhere else —
    no leakage).

    Fold balancing (two greedy heuristics — the alternative to a random split, which
    leaves the last group oversized and the continuous-viewing-zone stars piled into a
    few folds):
      1. SECTOR PARTITION by star-load (LPT): assign each sector — heaviest first — to
         the group with the least accumulated load (≈ number of stars observed in it).
         Equalizes how many stars get held out per fold → even TRAINING sizes.
      2. VALIDATION assignment greedily: process the most-constrained stars first
         (fewest touched groups — single-sector stars are locked) and send each to its
         least-loaded touched group → even VALIDATION sizes.
    Both are deterministic given `seed` (used only to break load ties in the partition).

    Args:
        star_sectors: list/array of iterables of sector numbers, one entry per star,
                      aligned to the aggregated latent rows.
        n_folds:      requested number of sector groups (capped at #unique sectors).
        seed:         RNG seed; only breaks ties in the load-balanced partition.

    Returns:
        fold_of_sample:    (N,) int — validation fold per star
        star_touch_groups: list of frozenset[int] — group indices each star touches
        n_folds_eff:       number of sector groups (== n_folds unless capped by #sectors)
    """
    # Per-sector load = number of stars observed in that sector.
    sec_load = {}
    for ss in star_sectors:
        for s in ss:
            s = int(s)
            sec_load[s] = sec_load.get(s, 0) + 1
    all_secs = sorted(sec_load)
    if not all_secs:
        raise ValueError('loso_sector_folds: no sectors found in star_sectors.')
    n_folds_eff = min(n_folds, len(all_secs))

    # (1) LPT load-balanced partition. Seed shuffles first so equal-load sectors get a
    # seed-dependent (but reproducible) tie order; the stable sort by -load keeps it.
    rng = np.random.RandomState(seed)
    order = [int(x) for x in rng.permutation(all_secs)]
    order.sort(key=lambda s: -sec_load[s])
    group_load = [0] * n_folds_eff
    sec_to_group = {}
    for s in order:
        f = int(np.argmin(group_load))
        sec_to_group[s] = f
        group_load[f] += sec_load[s]

    star_touch_groups = [frozenset(sec_to_group[int(s)] for s in ss) for ss in star_sectors]

    # (2) Greedy validation assignment: most-constrained stars first, each to its
    # least-loaded touched group. Ties resolved by index/group order (deterministic).
    val_count = [0] * n_folds_eff
    fold_of_sample = np.empty(len(star_sectors), dtype=int)
    for i in sorted(range(len(star_sectors)), key=lambda i: len(star_touch_groups[i])):
        f = min(sorted(star_touch_groups[i]), key=lambda g: val_count[g])
        fold_of_sample[i] = f
        val_count[f] += 1
    return fold_of_sample, star_touch_groups, n_folds_eff


def run_kfold_cv(latent_vectors, ages, bprp0, bprp0_err, mg, mem_prob, tic_ids,
                 n_folds=10, pca_dim=32, pca_latents=None,
                 lr=1e-3, weight_decay=1e-4, n_epochs=100, batch_size=64,
                 device=None, seed=42, save_models_dir=None,
                 star_level_split=False,
                 flow_transforms=8, flow_hidden_features=None, loga_grid_size=None,
                 encoder_type='pca', mlp_encoder_hidden=None, aux_loss_weight=1.0,
                 use_mg=False, lr_decay_rate=None,
                 dropout=0.1, variance_reg_weight=0.0,
                 training_stages='joint', encoder_pretrain_epochs=100,
                 joint_finetune_epochs=50, finetune_encoder_lr_mult=0.01,
                 finetune_flow_lr_mult=0.1,
                 train_full=False, source=None,
                 skip_log10=False, age_grid_range=None,
                 age_err=None, k_age_samples=1,
                 prediction_mode='nle',
                 sectors=None,
                 sector_level_split=False, sector_split_drop_star_overlap=True,
                 balance_sector_age=False, n_balance_age_bins=10,
                 adv_sector_weight=0.0, adv_hidden=64,
                 loocv_age=False,
                 loso=False, loso_star_sectors=None, loso_strict_train=True,
                 prefit_pca_bundle=None):
    """Run k-fold cross-validation.

    Args:
        star_level_split: if True, folds are assigned by unique TIC ID so all
                          sectors of a star always land in the same fold. This
                          avoids data leakage when the same star has multiple sectors.
                          Set to True for 'predict_mean'; irrelevant for aggregated
                          methods where there is already one row per star.

    Returns:
        predictions, true_ages, tic_ids, fold_assignments, fold_losses, norm_params
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(seed)
    n_samples = len(ages)

    # Normalise latent vectors (X).  BPRP0 is kept separate — it goes to the
    # flow as context alongside log10_age, not into the MLP encoder.
    X = latent_vectors
    # `ages` are normally linear Myr, log10'd here for the flow context.
    # When skip_log10=True the input is treated as already-on-target-scale
    # (e.g. a pre-normalized z-score); the grid range must match.
    y = ages.astype(np.float32) if skip_log10 else np.log10(ages)

    # Per-star age uncertainty propagation. When k_age_samples > 1 we draw
    # K Gaussian samples around each star's central age (mean=y, std=age_err)
    # and stack them into an (N, K) array. The model averages NLL per-star
    # over the K samples before averaging across the batch — this avoids
    # weighting stars unequally by sample count and keeps the batch-level
    # loss on the same scale as the single-sample case.
    if k_age_samples > 1:
        if age_err is None:
            raise ValueError('k_age_samples > 1 requires age_err to be provided.')
        rng = np.random.default_rng(seed)
        sigma = np.where(np.isnan(age_err), 0.0, np.maximum(age_err, 0.0)).astype(np.float32)
        noise = rng.standard_normal(size=(len(y), k_age_samples)).astype(np.float32)
        y = (y[:, None] + sigma[:, None] * noise).astype(np.float32)
        n_with_err = int(np.sum(sigma > 0))
        print(f'Drew {k_age_samples} Gaussian age samples per star '
              f'({n_with_err}/{len(sigma)} have non-zero σ; rest get K identical copies).')

    # Per-star central age (1D), for printable metrics that compare against
    # the model's point-estimate predictions.
    y_central = y if y.ndim == 1 else y.mean(axis=1)

    # Pre-fit PCA artifact (computed once on the full pool, reused across the dim
    # sweep + LOSO) takes priority over the inline `pca_latents`-based fit below.
    if prefit_pca_bundle is not None:
        global_pca, X_mean, X_std = prefit_pca_bundle
        if encoder_type == 'pca' and global_pca.n_components_ != pca_dim:
            raise ValueError(
                f'prefit_pca_bundle has {global_pca.n_components_} components but '
                f'--pca_dim={pca_dim}; truncate at load time so they match.')
        print(f'Normalization + PCA basis from prefit_pca_bundle '
              f'(no per-fold refit; identical basis across runs).')
    elif pca_latents is not None:
        X_mean, X_std = pca_latents.mean(axis=0), pca_latents.std(axis=0) + 1e-8
        print(f'Normalization computed from {len(pca_latents)} all-star latents')
    else:
        X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # bprp0: passed raw (~0–4 scale, matches log10_age range)
    # bprp0_err: log10 transform only
    # mg: passed raw (MG_quick is already a log quantity in [-8, +12]; double-log
    #     would clip negative values — i.e. all giants — to a fixed floor)
    bprp0_norm     = bprp0
    bprp0_err_norm = np.log10(np.clip(bprp0_err, 1e-10, None))
    mg_norm        = mg

    norm_params = {
        'X_mean': X_mean, 'X_std': X_std,
        'latent_dim': latent_vectors.shape[1],
        'input_dim': X.shape[1],
    }

    # Fit a single global PCA on all-star latents (if provided AND no prefit
    # bundle was passed in), so every fold uses the same coordinate system
    # rather than re-fitting per fold. When prefit_pca_bundle is set, global_pca
    # was already populated above.
    if prefit_pca_bundle is not None:
        pass  # global_pca already bound from the bundle
    else:
        global_pca = None
        if pca_latents is not None and encoder_type == 'pca':
            pca_norm   = (pca_latents - X_mean) / X_std
            global_pca = PCA(n_components=pca_dim, whiten=True)
            global_pca.fit(pca_norm)
            var_expl = global_pca.explained_variance_ratio_.sum()
            print(f'Global PCA {pca_dim}D fitted on {len(pca_latents)} stars: '
                  f'{100*var_expl:.1f}% variance explained')
            del pca_norm  # free the normalized copy; global_pca is now self-contained

    # Age grid for posterior inference
    G = loga_grid_size or LOGA_GRID_DEFAULT_SIZE
    grid_min, grid_max = age_grid_range if age_grid_range is not None else PRIOR_LOGA_MYR
    loga_grid_np = np.linspace(grid_min, grid_max, G)
    loga_grid_t  = torch.tensor(loga_grid_np, dtype=torch.float32)

    # Sector handling for the disjoint-CV / balancing / adversary options.
    # All three need a per-sample sector, which only exists for the per-sector
    # aggregations ('none' / 'predict_mean'); aggregated modes collapse sectors.
    needs_sector = sector_level_split or balance_sector_age or adv_sector_weight > 0
    if needs_sector and sectors is None:
        raise ValueError(
            'sector_level_split / balance_sector_age / adv_sector_weight require '
            'per-sample `sectors` — run with --star_aggregation none (or predict_mean).')
    sec_arr = np.asarray(sectors) if sectors is not None else None
    if sec_arr is not None:
        uniq_sec    = np.unique(sec_arr)
        sec_to_idx  = {int(s): i for i, s in enumerate(uniq_sec)}
        sec_idx_all = np.array([sec_to_idx[int(s)] for s in sec_arr], dtype=np.int64)
        n_sectors   = len(uniq_sec)
    else:
        sec_idx_all, n_sectors = None, 0

    # Build fold assignments
    fold_of_sample = np.zeros(n_samples, dtype=int)

    combined_mode = source is not None
    if combined_mode and sector_level_split:
        print('  NOTE: combined_mode + sector_level_split — using sector split (combined ignored).')
        combined_mode = False
    if loocv_age:
        # Leave-one-cluster-out via the shared age-grouping (young bundled, big
        # clusters individual, sparse ages bundled). n_folds is OVERRIDDEN. Each
        # held-out fold spans only held-out ages, so read the GLOBAL r/MAE over all
        # held-out predictions (plot_kfold_results does this), not per-fold r.
        fold_of_sample, n_folds = loocv_age_folds(ages)
        print(f'Leave-one-cluster-out (age-grouped, ChronoFlow proxy): {n_folds} folds')
    elif loso:
        # Leave-one-sector-out (per-star). Partition unique sectors into n_folds
        # groups; each star validates in one fold (a group its sectors touch) and is
        # excluded from training in EVERY fold whose group it touches. star_touch_groups
        # is used in the loop below to build a per-fold star-disjoint training set.
        if loso_star_sectors is None:
            raise ValueError('loso=True requires loso_star_sectors (per-star sector sets).')
        if len(loso_star_sectors) != n_samples:
            raise ValueError(
                f'loso_star_sectors length {len(loso_star_sectors)} != n_samples {n_samples} '
                '(must be one sector-set per aggregated star row).')
        fold_of_sample, star_touch_groups, n_folds = loso_sector_folds(
            loso_star_sectors, n_folds=n_folds, seed=seed)
        n_secs = len({int(s) for ss in loso_star_sectors for s in ss})
        print(f'Leave-one-sector-out: {n_secs} unique sectors → {n_folds} disjoint groups; '
              f'{n_samples} stars. Training drops every star touching the held-out group.')
    elif sector_level_split:
        # Hold out entire SECTORS per fold so each fold predicts stars observed
        # in sectors absent from training — a proxy for field-star generalization.
        uniq          = np.unique(sec_arr)
        perm          = np.random.permutation(len(uniq))
        secs_shuffled = uniq[perm]
        fold_size     = max(len(secs_shuffled) // n_folds, 1)
        sec_fold      = {}
        for f in range(n_folds):
            s = f * fold_size
            e = s + fold_size if f < n_folds - 1 else len(secs_shuffled)
            for sv in secs_shuffled[s:e]:
                sec_fold[int(sv)] = f
        fold_of_sample = np.array([sec_fold[int(s)] for s in sec_arr])
        print(f'Sector-disjoint split: {len(uniq)} unique sectors across {n_folds} folds '
              f'(star-overlap drop={sector_split_drop_star_overlap}).')
    elif combined_mode:
        # K-fold over host stars only; non-host (source==0) rows always train
        host_mask    = (source == 1)
        host_idx_all = np.where(host_mask)[0]
        host_tics    = tic_ids[host_mask]
        unique_host_tics = np.unique(host_tics)
        perm             = np.random.permutation(len(unique_host_tics))
        tics_shuffled    = unique_host_tics[perm]
        tic_fold_size    = max(len(tics_shuffled) // n_folds, 1)
        tic_to_fold = {}
        for f in range(n_folds):
            s = f * tic_fold_size
            e = s + tic_fold_size if f < n_folds - 1 else len(tics_shuffled)
            for t in tics_shuffled[s:e]:
                tic_to_fold[t] = f

        fold_of_sample = np.full(n_samples, -1, dtype=int)
        for i in host_idx_all:
            fold_of_sample[i] = tic_to_fold[tic_ids[i]]
        print(f'Combined k-fold split: {len(unique_host_tics)} unique hosts across '
              f'{n_folds} folds; {(source == 0).sum()} non-host rows always in train.')
    elif star_level_split:
        unique_tics   = np.unique(tic_ids)
        perm          = np.random.permutation(len(unique_tics))
        tics_shuffled = unique_tics[perm]
        tic_fold_size = len(tics_shuffled) // n_folds
        tic_to_fold   = {}
        for f in range(n_folds):
            s = f * tic_fold_size
            e = s + tic_fold_size if f < n_folds - 1 else len(tics_shuffled)
            for t in tics_shuffled[s:e]:
                tic_to_fold[t] = f
        fold_of_sample = np.array([tic_to_fold[t] for t in tic_ids])
        print(f'Star-level split: {len(unique_tics)} unique stars across {n_folds} folds')
    else:
        perm      = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds
        for f in range(n_folds):
            s = f * fold_size
            e = s + fold_size if f < n_folds - 1 else n_samples
            fold_of_sample[perm[s:e]] = f

    # One array per posterior statistic (all in log10_age space)
    stat_keys = ('median', 'mean', 'map', 'p16', 'p84')
    # NaN init: in sector-disjoint mode some rows are intentionally never predicted
    # (star-overlap drop). Normal modes predict every row, so no NaN survives.
    all_stats   = {k: np.full(n_samples, np.nan) for k in stat_keys}
    fold_losses = []
    fold_models = []

    print(f'\nRunning {n_folds}-fold CV on {n_samples} samples '
          f'({"star-level" if star_level_split else "sample-level"} split)...')

    for fold in range(n_folds):
        val_idx = np.where(fold_of_sample == fold)[0]
        if combined_mode:
            # Non-host rows (fold_of_sample == -1) join train every fold
            train_idx = np.where((source == 0) | ((source == 1) & (fold_of_sample != fold)))[0]
        elif loso:
            # Validation = stars assigned to fold f (touch its sector group).
            # Training mode controlled by loso_strict_train:
            #   True  (STRICT, default): train_idx = stars whose sectors are ALL outside
            #         the held-out group. Excludes every star touching a held-out sector
            #         (the whole star, not just its held-out-sector rows). The most
            #         aggressive sector-disjoint test — also the most data-removing.
            #   False (RELAXED): train_idx = all stars NOT in val_idx. Same val partition
            #         as strict (stratified-by-sector-touching), but training keeps stars
            #         that touch held-out sectors. At PCA dim 4 with a smooth flow there's
            #         no per-star memorization, so this is a fair test of "sector-aware
            #         validation, full training context".
            if loso_strict_train:
                train_idx = np.array(
                    [i for i in range(n_samples) if fold not in star_touch_groups[i]], dtype=int)
            else:
                train_idx = np.where(fold_of_sample != fold)[0]
            if len(val_idx) == 0:
                print(f'    Fold {fold + 1}: no stars assigned to this sector group; skipping.')
                continue
        else:
            train_idx = np.where(fold_of_sample != fold)[0]

        # Sector-disjoint CV: optionally drop val rows whose star ALSO appears in a
        # training sector, so the held-out evaluation is star-disjoint too (a true
        # "unseen star in an unseen sector" estimate). Dropped rows stay NaN.
        if sector_level_split and sector_split_drop_star_overlap:
            train_tics = set(int(t) for t in tic_ids[train_idx])
            keep = np.array([int(t) not in train_tics for t in tic_ids[val_idx]], dtype=bool)
            n_drop = int((~keep).sum())
            val_idx = val_idx[keep]
            if n_drop:
                print(f'    Fold {fold + 1}: dropped {n_drop} val rows whose star also '
                      f'appears in a training sector (star-disjoint).')
            if len(val_idx) == 0:
                print(f'    Fold {fold + 1}: no star-disjoint val rows left; skipping fold.')
                continue

        sec_idx_train = sec_idx_all[train_idx] if sec_idx_all is not None else None

        X_train, y_train   = X_norm[train_idx],        y[train_idx]
        X_val              = X_norm[val_idx]
        bprp0_train        = bprp0_norm[train_idx]
        bprp0_val          = bprp0_norm[val_idx]
        bprp0_err_train    = bprp0_err_norm[train_idx]
        bprp0_err_val      = bprp0_err_norm[val_idx]
        mg_train           = mg_norm[train_idx]
        mg_val             = mg_norm[val_idx]
        mem_prob_train     = mem_prob[train_idx]
        mem_prob_val_fold  = mem_prob[val_idx]

        val_stats, train_loss, model_state, fold_pca, train_losses, val_losses = train_single_fold(
            X_train, y_train, bprp0_train, bprp0_err_train, mg_train, mem_prob_train,
            X_val, y[val_idx], bprp0_val, bprp0_err_val, mg_val, mem_prob_val_fold,
            pca_dim, lr, weight_decay, n_epochs, batch_size, device,
            flow_transforms=flow_transforms, flow_hidden_features=flow_hidden_features,
            loga_grid=loga_grid_t,
            encoder_type=encoder_type, mlp_encoder_hidden=mlp_encoder_hidden,
            aux_loss_weight=aux_loss_weight, use_mg=use_mg,
            lr_decay_rate=lr_decay_rate,
            global_pca=global_pca,
            dropout=dropout, variance_reg_weight=variance_reg_weight,
            training_stages=training_stages,
            encoder_pretrain_epochs=encoder_pretrain_epochs,
            joint_finetune_epochs=joint_finetune_epochs,
            finetune_encoder_lr_mult=finetune_encoder_lr_mult,
            finetune_flow_lr_mult=finetune_flow_lr_mult,
            prediction_mode=prediction_mode,
            sectors_idx_train=sec_idx_train, n_sectors=n_sectors,
            balance_sector_age=balance_sector_age, n_balance_age_bins=n_balance_age_bins,
            adv_sector_weight=adv_sector_weight, adv_hidden=adv_hidden,
        )
        fold_models.append({
            'state_dict': model_state, 'pca': fold_pca,
            'train_losses': train_losses, 'val_losses': val_losses,
        })

        for k in stat_keys:
            all_stats[k][val_idx] = val_stats[k]
        fold_of_sample[val_idx] = fold  # already set, but explicit
        fold_losses.append(train_loss)

        val_pred_log = val_stats['median']
        val_mae  = np.mean(np.abs(val_pred_log - y_central[val_idx]))  # dex
        # In leave-one-age-out a fold holds a single age (no variance) → r undefined.
        val_corr = (np.corrcoef(val_pred_log, y_central[val_idx])[0, 1]
                    if np.std(y_central[val_idx]) > 0 else np.nan)
        print(f'  Fold {fold + 1}/{n_folds}: best_val_nll={train_loss:.4f}  '
              f'val_MAE={val_mae:.3f} dex  val_r={val_corr:.3f}')

    if save_models_dir is not None:
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # n_eff < n_folds when sector-disjoint CV skips folds that have no
        # star-disjoint val rows; index by the actual fold_models length.
        n_eff = len(fold_models)
        torch.save({
            'fold_models': fold_models,   # list of {'state_dict': ..., 'pca': sklearn.PCA, 'train_losses': ..., 'val_losses': ...}
            'normalization_params': norm_params,
            'pca_dim': pca_dim,
            'n_folds': n_eff,
            'seed': seed,
        }, save_dir / 'kfold_models.pt')
        print(f'Saved {n_eff} fold models to {save_dir / "kfold_models.pt"}'
              + (f' ({n_folds - n_eff} fold(s) skipped — no star-disjoint val rows)'
                 if n_eff < n_folds else ''))

        import json
        loss_curves = {
            f'fold_{f}': {
                'train_nll': fold_models[f]['train_losses'],
                'val_nll':   fold_models[f]['val_losses'],
            }
            for f in range(n_eff)
        }
        with open(save_dir / 'training_curves.json', 'w') as fp:
            json.dump(loss_curves, fp)
        print(f'Saved training curves to {save_dir / "training_curves.json"}')

    # ── Optional: train a single model on ALL labeled stars for deployment ──
    if train_full and save_models_dir is not None:
        print('\n' + '='*60)
        print('Training full model on ALL labeled stars for deployment')
        print('='*60)
        save_dir = Path(save_models_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        full_stats, full_loss, full_state, full_pca, full_train, full_val = train_single_fold(
            X_norm, y, bprp0_norm, bprp0_err_norm, mg_norm, mem_prob,   # train = all
            X_norm, y, bprp0_norm, bprp0_err_norm, mg_norm, mem_prob,   # val = all (tracking only)
            pca_dim, lr, weight_decay, n_epochs, batch_size, device,
            flow_transforms=flow_transforms, flow_hidden_features=flow_hidden_features,
            loga_grid=loga_grid_t,
            encoder_type=encoder_type, mlp_encoder_hidden=mlp_encoder_hidden,
            aux_loss_weight=aux_loss_weight, use_mg=use_mg,
            lr_decay_rate=lr_decay_rate,
            global_pca=global_pca,
            dropout=dropout, variance_reg_weight=variance_reg_weight,
            training_stages=training_stages,
            encoder_pretrain_epochs=encoder_pretrain_epochs,
            joint_finetune_epochs=joint_finetune_epochs,
            finetune_encoder_lr_mult=finetune_encoder_lr_mult,
            finetune_flow_lr_mult=finetune_flow_lr_mult,
            prediction_mode=prediction_mode,
            sectors_idx_train=sec_idx_all, n_sectors=n_sectors,
            balance_sector_age=balance_sector_age, n_balance_age_bins=n_balance_age_bins,
            adv_sector_weight=adv_sector_weight, adv_hidden=adv_hidden,
        )

        torch.save({
            'fold_models': [{'state_dict': full_state, 'pca': full_pca or global_pca,
                             'train_losses': full_train, 'val_losses': full_val}],
            'normalization_params': norm_params,
            'pca_dim': pca_dim,
            'n_folds': 1,
            'seed': seed,
        }, save_dir / 'full_model.pt')

        full_pred_log = full_stats['median']
        if combined_mode:
            host_eval = (source == 1)
            full_mae_h  = np.mean(np.abs(full_pred_log[host_eval] - y_central[host_eval]))
            full_corr_h = np.corrcoef(full_pred_log[host_eval], y_central[host_eval])[0, 1]
            print(f'Full model: train_nll={full_loss:.4f}  '
                  f'host MAE={full_mae_h:.3f} dex  host r={full_corr_h:.3f}')
        else:
            full_mae  = np.mean(np.abs(full_pred_log - y_central))
            full_corr = np.corrcoef(full_pred_log, y_central)[0, 1]
            print(f'Full model: train_nll={full_loss:.4f}  '
                  f'MAE={full_mae:.3f} dex  r={full_corr:.3f}')
        print(f'Saved full model to {save_dir / "full_model.pt"}')

    all_predictions = all_stats['median']  # log10(age/Myr), used as primary prediction
    return all_predictions, all_stats, ages, tic_ids, fold_of_sample, fold_losses, norm_params


# ---------------------------------------------------------------------------
# Results / plotting
# ---------------------------------------------------------------------------

def plot_kfold_results(predictions, true_ages, bprp0, tic_ids, fold_assignments, output_dir,
                       pred_stats=None):
    """Create diagnostic plots and save metrics + predictions CSV.

    predictions: log10(age/Myr) — median from flow posterior
    true_ages:   linear Myr (raw data)
    pred_stats:  dict of log10(age/Myr) arrays: median, mean, map, p16, p84
    """
    # Drop rows without a prediction (sector-disjoint star-overlap leaves NaNs);
    # normal modes predict every row, so this is a no-op there.
    predictions = np.asarray(predictions)
    true_ages   = np.asarray(true_ages)
    finite = np.isfinite(predictions) & np.isfinite(true_ages) & (true_ages > 0)
    if not finite.all():
        n_drop = int((~finite).sum())
        print(f'  [metrics] excluding {n_drop} row(s) without a finite prediction '
              f'(e.g. sector-disjoint star-overlap drops).')
        predictions = predictions[finite]
        true_ages   = true_ages[finite]
        bprp0       = np.asarray(bprp0)[finite]
        tic_ids     = np.asarray(tic_ids)[finite]
        fold_assignments = np.asarray(fold_assignments)[finite]
        if pred_stats is not None:
            pred_stats = {k: np.asarray(v)[finite] for k, v in pred_stats.items()}

    log_true  = np.log10(true_ages)
    log_resid = predictions - log_true   # dex (pred − true)
    mae       = np.mean(np.abs(log_resid))
    rmse      = np.sqrt(np.mean(log_resid ** 2))
    med_ae    = np.median(np.abs(log_resid))
    bias      = np.mean(log_resid)
    scat      = np.std(log_resid)
    corr      = np.corrcoef(predictions, log_true)[0, 1]

    print(f'\n=== K-Fold Results ===')
    print(f'N = {len(true_ages)}')
    print(f'MAE = {mae:.3f} dex   Median AE = {med_ae:.3f} dex   RMSE = {rmse:.3f} dex')
    print(f'Bias = {bias:.3f} dex   Scatter = {scat:.3f} dex   Pearson r = {corr:.3f}')

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    sc  = ax1.scatter(log_true, predictions, c=bprp0, cmap='coolwarm',
                      alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    lo, hi = log_true.min(), log_true.max()
    ax1.plot([lo, hi], [lo, hi], 'k--', lw=2, label='Perfect')
    ax1.plot([lo, hi], [lo + 0.3, hi + 0.3], 'k:', alpha=0.5, label='±0.3 dex')
    ax1.plot([lo, hi], [lo - 0.3, hi - 0.3], 'k:', alpha=0.5)
    ax1.set_xlabel('log₁₀(True Age / Myr)'); ax1.set_ylabel('log₁₀(Predicted Age / Myr)')
    ax1.set_title(f'K-Fold Age Predictions (r={corr:.3f})')
    ax1.legend(loc='upper left')
    plt.colorbar(sc, ax=ax1, label='BP-RP₀')
    ax1.text(0.02, 0.98,
             f'N={len(true_ages)}\nMAE={mae:.3f} dex\nσ={scat:.3f} dex\nr={corr:.3f}',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(log_true, log_resid, c=bprp0, cmap='coolwarm',
                alpha=0.6, s=15, vmin=0.5, vmax=2.0)
    ax2.axhline(0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('log₁₀(True Age / Myr)'); ax2.set_ylabel('Residual (dex)')
    ax2.set_title('Residuals vs True Age')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(log_resid, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', lw=2, label='Zero')
    ax3.axvline(bias, color='orange', lw=2, label=f'Bias={bias:.3f}')
    ax3.set_xlabel('Residual (dex)'); ax3.set_ylabel('Count')
    ax3.set_title(f'Residuals (σ={scat:.3f} dex)')
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(bprp0, log_resid, c=log_true, cmap='viridis', alpha=0.6, s=15)
    ax4.axhline(0, color='k', linestyle='--', lw=2)
    ax4.set_xlabel('BP-RP₀'); ax4.set_ylabel('Residual (dex)')
    ax4.set_title('Residuals vs Colour (by log age)')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(log_true.min(), log_true.max()))
    plt.colorbar(sm, ax=ax4, label='log₁₀(Age / Myr)')

    plt.tight_layout()
    plt.savefig(output_dir / 'kfold_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {output_dir / "kfold_results.png"}')

    df = pd.DataFrame({
        'TIC_ID':             tic_ids,
        'log10_true_age':     log_true,
        'log10_pred_median':  predictions,
        'residual_dex':       log_resid,
        'bprp0':              bprp0,
        'fold':               fold_assignments,
    })
    if pred_stats is not None:
        df['log10_pred_mean'] = pred_stats['mean']
        df['log10_pred_map']  = pred_stats['map']
        df['log10_pred_p16']  = pred_stats['p16']
        df['log10_pred_p84']  = pred_stats['p84']
    df.to_csv(output_dir / 'kfold_predictions.csv', index=False)
    print(f'Saved predictions to {output_dir / "kfold_predictions.csv"}')

    metrics = {
        'n_samples':     int(len(true_ages)),
        'mae_dex':       float(mae),
        'median_ae_dex': float(med_ae),
        'rmse_dex':      float(rmse),
        'bias_dex':      float(bias),
        'scatter_dex':   float(scat),
        'correlation':   float(corr),
    }
    with open(output_dir / 'kfold_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='K-fold cross-validation for age prediction')

    # Data / model
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint. Required unless --load_latents is set.')
    parser.add_argument('--h5_path',       type=str, default='data/timeseries_x.h5')
    parser.add_argument('--age_csv',       type=str, default='data/phot_all.csv')
    parser.add_argument('--pca_h5_paths',  type=str, nargs='+', default=None,
                        help='H5 files to use for global PCA/normalization (all stars, no age filter). '
                             'If omitted, PCA is fit per-fold on training stars only.')
    parser.add_argument('--output_dir', type=str, default='output/age_predictor/kfold')

    # (kept for backward compat with old shell scripts; not used)
    parser.add_argument('--pickle_path', type=str, default=None,
                        help='Deprecated — ignored. Data is now loaded from --h5_path.')

    # Model extraction
    parser.add_argument('--hidden_size',      type=int,   default=64)
    parser.add_argument('--direction',        type=str,   default='bi',
                        choices=['forward', 'backward', 'bi'])
    parser.add_argument('--mode',             type=str,   default='parallel',
                        choices=['sequential', 'parallel'])
    parser.add_argument('--use_flow',         action='store_true')
    parser.add_argument('--use_metadata',     action='store_true')
    parser.add_argument('--use_conv_channels',action='store_true')
    parser.add_argument('--conv_encoder_type',type=str,   default='unet',
                        choices=['lightweight', 'unet'])
    parser.add_argument('--max_length',       type=int,   default=16384)
    parser.add_argument('--pooling_mode',     type=str,   default='multiscale',
                        choices=['mean', 'multiscale', 'final'])

    # Star aggregation
    parser.add_argument('--star_aggregation', type=str,   default='none',
                        choices=['none', 'predict_mean', 'latent_mean', 'latent_median',
                                 'latent_max', 'latent_mean_std', 'cross_sector'],
                        help=(
                            'none          – one sample per sector; kfold splits by sector '
                            '(legacy, has leakage for multi-sector stars). '
                            'predict_mean  – kfold splits by star; per-sector predictions '
                            'are averaged per star before metrics. '
                            'latent_mean   – sector latents are mean-pooled per star. '
                            'latent_median – sector latents are median-pooled per star. '
                            'cross_sector  – hidden states concatenated across sectors '
                            'before multiscale pooling.'
                        ))

    # Data selection
    parser.add_argument('--max_samples',     type=int,   default=None)
    parser.add_argument('--stratify_by_age', action='store_true')
    parser.add_argument('--n_age_bins',      type=int,   default=20)
    parser.add_argument('--bprp0_min',       type=float, default=None)
    parser.add_argument('--bprp0_max',       type=float, default=None)
    parser.add_argument('--require_prot',    action='store_true')

    # K-fold
    parser.add_argument('--n_folds',    type=int,   default=10)
    parser.add_argument('--seed',       type=int,   default=42)

    # Sector-confound mitigations (all need per-sample sectors -> use
    # --star_aggregation none or predict_mean).
    parser.add_argument('--sector_level_split', action='store_true',
                        help='Hold out whole SECTORS per fold (sector-disjoint CV) so each '
                             'fold predicts stars from unseen sectors — a field-star '
                             'generalization proxy. Replaces the star/sample split.')
    parser.add_argument('--sector_split_keep_star_overlap', action='store_true',
                        help='With --sector_level_split, KEEP val rows whose star also '
                             'appears in a training sector (default: drop them so the '
                             'held-out set is star-disjoint too).')
    parser.add_argument('--balance_sector_age', action='store_true',
                        help='Sample training rows to flatten the (sector x age-bin) joint, '
                             'so over-represented cluster cells stop dominating and the model '
                             'cannot lean on sector->age.')
    parser.add_argument('--n_balance_age_bins', type=int, default=10,
                        help='Number of age bins for --balance_sector_age (default 10).')
    parser.add_argument('--adv_sector_weight', type=float, default=0.0,
                        help='Peak GRL coefficient for the sector adversary on the MLP '
                             'bottleneck (0 = off). Ramped 0->this over training. Tune on '
                             'the Pareto front: sector probe down while partial r stays up. '
                             'Requires --encoder_type mlp/linear.')
    parser.add_argument('--adv_hidden', type=int, default=64,
                        help='Hidden width of the GRL sector-adversary MLP (default 64).')
    parser.add_argument('--override_ages_from_csv', action='store_true',
                        help='In --load_latents mode, re-join ages from --age_csv by GaiaDR3_ID, '
                             'overriding the cache ages. Use for ChronoFlow LOCO so grouping+target '
                             'use the clean ChronoFlow isochrone ages (metadata.csv), not the blended '
                             'multi-catalog ages baked into the cache. Rows not in the CSV are dropped.')
    parser.add_argument('--loocv_age', action='store_true',
                        help='Leave-one-age-out CV: each unique age value is its own fold, held '
                             'out one at a time (a cluster proxy for ChronoFlow, where each cluster '
                             'has one isochrone age). Overrides --n_folds. Tests cluster '
                             'generalization / anti-memorization. Read the GLOBAL r/MAE (per-fold r '
                             'is undefined since each fold is a single age).')
    parser.add_argument('--loso', action='store_true',
                        help='Leave-one-sector-out CV (per-star, --star_aggregation latent_max): '
                             'partition unique TESS sectors into --n_folds disjoint groups; each '
                             'star validates in one fold (a group its sectors touch) and is removed '
                             'from training in every fold whose group it touches (the whole star, not '
                             'just its held-out-sector rows). Estimates generalization to stars '
                             'observed only in sectors the age model never trained on. Read the '
                             'GLOBAL r/MAE. Mutually exclusive with --loocv_age / --sector_level_split.')
    parser.add_argument('--loso_relaxed_train', action='store_true',
                        help='With --loso, RELAX the training-set rule: keep stars that touch '
                             'held-out sectors in training (training = all stars NOT in val), '
                             'rather than the strict default that excludes every touching star. '
                             'Validation partition is unchanged. Tests sector-aware validation '
                             'with full training context; sound at small PCA dims where per-star '
                             'memorization is precluded by the smooth-flow-over-low-D-bottleneck.')

    # Age predictor training
    parser.add_argument('--encoder_type',         type=str,   default='pca',
                        choices=['pca', 'mlp', 'linear'],
                        help='Compression method: pca (fixed, per-fold), mlp (learned, '
                             'jointly trained with auxiliary age loss), or linear (learned '
                             'linear projection + LayerNorm, comparable to PCA). Default: pca')
    parser.add_argument('--pca_dim',              type=int,   default=32,
                        help='Bottleneck dimensionality for both pca and mlp encoders (default: 32)')
    parser.add_argument('--mlp_encoder_hidden',   type=int,   nargs='+', default=[256, 128],
                        help='Hidden layer sizes for the MLP encoder (mlp mode only, default: 256 128)')
    parser.add_argument('--aux_loss_weight',      type=float, default=1.0,
                        help='Weight for the auxiliary age L1 loss in mlp mode (default: 1.0)')
    parser.add_argument('--dropout',              type=float, default=0.1,
                        help='Dropout rate in MLP encoder layers (mlp mode only, default: 0.1)')
    parser.add_argument('--variance_reg_weight',  type=float, default=0.0,
                        help='VICReg variance regularization weight on bottleneck (mlp mode only, default: 0.0)')
    parser.add_argument('--training_stages',       type=str,   default='joint',
                        choices=['joint', 'two_stage', 'three_stage'],
                        help='Training mode: joint (all params together), two_stage (encoder '
                             'then flow), three_stage (encoder, flow, joint fine-tune). '
                             'Staged modes only apply to mlp/linear encoders. Default: joint')
    parser.add_argument('--encoder_pretrain_epochs', type=int, default=100,
                        help='Epochs for encoder pre-training in two_stage/three_stage mode (default: 100)')
    parser.add_argument('--joint_finetune_epochs',   type=int, default=50,
                        help='Epochs for joint fine-tuning in three_stage mode (default: 50)')
    parser.add_argument('--finetune_encoder_lr_mult', type=float, default=0.01,
                        help='Encoder LR multiplier during stage 3 fine-tuning, relative to '
                             'flow LR. E.g. 0.01 = encoder trains at 1%% of flow LR. (default: 0.01)')
    parser.add_argument('--finetune_flow_lr_mult', type=float, default=0.1,
                        help='Flow LR multiplier at the start of stage 3, relative to the '
                             'original peak --lr. Stage 2 ended near LR=0 due to cosine '
                             'annealing; bumping back up to lr*mult gives stage 3 headroom '
                             'to fine-tune without the LR shock that comes from resetting '
                             'to peak. (default: 0.1)')
    parser.add_argument('--use_mg',               action='store_true', default=False,
                        help='Include MG_quick as a 4th flow context variable (default: off). '
                             'MG is passed raw (like BPRP0); MG is NOT included in this mode.')
    parser.add_argument('--use_mg_only',          action='store_true', default=False,
                        help='Use (MG, MG_err) instead of (BPRP0, BPRP0_err) as flow context. '
                             'Implemented as a swap before training so the flow still sees a '
                             '3-context input. Mutually exclusive with --use_mg.')

    # Subset filtering
    parser.add_argument('--subset_col', type=str, default=None,
                        help='Column name to filter on (e.g. "ref"). Applied after loading '
                             'latents; keeps only rows whose value is in --subset_val.')
    parser.add_argument('--subset_val', type=str, nargs='+', default=None,
                        help='Allowed value(s) for --subset_col (e.g. "ChronoFlow"). '
                             'Multiple values = OR logic.')
    parser.add_argument('--subset_csv', type=str, default=None,
                        help='CSV file containing --subset_col, joined on GaiaDR3_ID. '
                             'Defaults to --age_csv if omitted.')
    parser.add_argument('--lr',                   type=float, default=1e-3)
    parser.add_argument('--lr_decay_rate',        type=float, default=None,
                        help='Exponential decay rate per epoch for the LR scheduler. '
                             'Default: 10^(-3/n_epochs), which drops LR by 1000x over training. '
                             'Use e.g. 0.99 for a much shallower decay.')
    parser.add_argument('--weight_decay',         type=float, default=1e-4)
    parser.add_argument('--n_epochs',             type=int,   default=100)
    parser.add_argument('--batch_size',           type=int,   default=64)
    parser.add_argument('--flow_transforms',      type=int,   default=8,
                        help='Number of NSF transforms in the flow head (default: 8)')
    parser.add_argument('--flow_hidden_dims',     type=int,   nargs='+', default=[64, 64],
                        help='Hidden layer sizes inside each flow transform (default: 64 64)')
    parser.add_argument('--loga_grid_size',       type=int,   default=LOGA_GRID_DEFAULT_SIZE,
                        help='Number of age grid points for posterior inference (default: 1000)')

    # Full-training model
    parser.add_argument('--train_full',   action='store_true',
                        help='After k-fold CV, train a final model on ALL labeled stars '
                             'and save as full_model.pt for deployment to unlabeled data.')

    # Latents cache
    parser.add_argument('--save_latents', type=str, default=None, metavar='PATH',
                        help='After extracting latents, save them (+ cross-sector latents) '
                             'to this npz file. Pass to --load_latents on subsequent runs '
                             'to skip model inference entirely.')
    parser.add_argument('--load_latents', type=str, default=None, metavar='PATH',
                        help='Load pre-computed latents from this npz file instead of '
                             'running the model. --model_path is not required when this '
                             'flag is set.')
    parser.add_argument('--pca_latent_pool', type=str, nargs='+', default=None, metavar='PATH',
                        help='In --load_latents mode, list of npz cache paths whose '
                             'latent_vectors are concatenated to form the GLOBAL PCA fit pool '
                             '(the basis used by every fold, instead of refitting per-fold on '
                             'each X_train). Recommended: all three cache files '
                             '(pretrain+hosts+thickdisk) so the basis spans the full encoder '
                             'output space — not just the labeled-cluster subset. Critical for '
                             'LOSO, where the per-fold training set systematically excludes '
                             'CVZ-touching stars and a per-fold PCA basis would itself be OOD.')
    parser.add_argument('--pca_cache', type=str, default=None, metavar='PATH',
                        help='Path to a persisted global PCA artifact (npz with X_mean/X_std + '
                             'PCA components). If the file EXISTS, load it (truncated to '
                             '--pca_dim) and skip pca_latents loading + the in-script PCA fit — '
                             'every run uses bit-identical normalization + basis. If it does NOT '
                             'exist, require --pca_latent_pool, fit a PCA at --pca_cache_max_dim '
                             'on the pool, save to this path, then use it (truncated to --pca_dim). '
                             'Lets the sweep + LOSO share one computed-once artifact.')
    parser.add_argument('--pca_cache_max_dim', type=int, default=16,
                        help='When --pca_cache file is being CREATED, fit and save this many '
                             'PCA components (default 16). Any subsequent run can request '
                             '--pca_dim <= this value and get the first --pca_dim components.')

    # Combined pretrain + host k-fold mode
    parser.add_argument('--combined_kfold', action='store_true',
                        help='Combined mode: train on (all non-host pretrain stars) ∪ '
                             '((K-1)/K of hosts), validate on held-out host fold. '
                             'Requires --host_h5_path and host data (or host cache).')
    parser.add_argument('--host_h5_path',  type=str, default=None,
                        help='H5 file containing host star light curves (combined mode).')
    parser.add_argument('--host_age_csv',  type=str, default=None,
                        help='CSV with host ages in Gyr (e.g. archive_ages_default.csv); '
                             'columns: GaiaDR3_ID, st_age, st_ageerr1, st_ageerr2. '
                             'Converted to Myr at load.')
    parser.add_argument('--host_metadata_csv', type=str, default=None,
                        help='CSV with host BPRP0 / BPRP0_err (joined onto host ages by '
                             'GaiaDR3_ID). Required if host_age_csv lacks photometry.')
    parser.add_argument('--save_latents_pretrain', type=str, default=None, metavar='PATH',
                        help='Cache path for non-host pretrain latents (combined mode).')
    parser.add_argument('--load_latents_pretrain', type=str, default=None, metavar='PATH',
                        help='Load non-host pretrain latents from cache (combined mode).')
    parser.add_argument('--save_latents_hosts', type=str, default=None, metavar='PATH',
                        help='Cache path for host latents (combined mode).')
    parser.add_argument('--load_latents_hosts', type=str, default=None, metavar='PATH',
                        help='Load host latents from cache (combined mode).')

    args = parser.parse_args()

    if args.use_mg and args.use_mg_only:
        parser.error('--use_mg and --use_mg_only are mutually exclusive.')

    if args.loso:
        if args.loocv_age or args.sector_level_split:
            parser.error('--loso is mutually exclusive with --loocv_age / --sector_level_split.')
        if args.star_aggregation != 'latent_max':
            parser.error('--loso currently supports --star_aggregation latent_max only '
                         '(per-star holdout). Got --star_aggregation '
                         f'{args.star_aggregation}.')

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}')

    conv_config = None
    if args.use_conv_channels:
        if args.conv_encoder_type == 'lightweight':
            conv_config = {'encoder_type': 'lightweight', 'hidden_channels': 16,
                           'num_layers': 3, 'activation': 'gelu'}
        else:
            conv_config = {'encoder_type': 'unet', 'input_length': 2500,
                           'encoder_dims': [4, 8, 16, 32], 'num_layers': 4,
                           'activation': 'gelu'}

    # ── Step 1: latents (from model or from cache) ──────────────────────────
    cached_cross_sector_latents = None
    cached_cross_sector_tic_ids = None
    pca_latents = None
    source = None  # set in combined_kfold mode: 0 = non-host pretrain, 1 = host

    def _fill_from_csv(gaia_ids, csv_path, col):
        df = pd.read_csv(csv_path)
        df['GaiaDR3_ID'] = df['GaiaDR3_ID'].astype(str)
        if col not in df.columns:
            return np.full(len(gaia_ids), np.nan, dtype=float)
        m = dict(zip(df['GaiaDR3_ID'], df[col]))
        return np.array([m.get(g, np.nan) for g in gaia_ids], dtype=float)

    def _load_cache_with_csv_fallback(cache_path, csv_path):
        c = load_latents_cache(cache_path)
        if c['bprp0_err'] is None:
            c['bprp0_err'] = _fill_from_csv(c['gaia_ids'], csv_path, 'BPRP0_err')
        if c['mg'] is None:
            c['mg'] = _fill_from_csv(c['gaia_ids'], csv_path, 'MG_quick')
        if c['mg_err'] is None:
            c['mg_err'] = _fill_from_csv(c['gaia_ids'], csv_path, 'MG_quick_err')
        if c['mem_prob'] is None:
            c['mem_prob'] = _fill_from_csv(c['gaia_ids'], csv_path, 'mem_prob_val')

        # Caches written by plot_umap_latent.py include unaged stars (UMAP is fit
        # on the full set). The fresh-extraction path filters these in load_ages;
        # mirror that here so the kfold trainer never sees NaN labels.
        ok = ~np.isnan(c['ages']) & ~np.isnan(c['bprp0']) & ~np.isnan(c['bprp0_err'])
        n_drop = (~ok).sum()
        if n_drop:
            print(f'  dropped {n_drop}/{len(ok)} rows with NaN age/bprp0/bprp0_err')
            for k in ('latent_vectors', 'ages', 'bprp0', 'bprp0_err',
                     'mg', 'mg_err', 'mem_prob', 'gaia_ids', 'tic_ids', 'sectors'):
                if c.get(k) is not None:
                    c[k] = c[k][ok]
        return c

    def _load_or_extract_one_source(label, h5_path, csv_path, save_path, load_path,
                                    age_loader_fn=None, csv_for_fallback=None):
        """Returns dict with the same keys as load_latents_cache plus 'pca_latents'."""
        if load_path:
            print(f'\n=== Loading {label} latents from cache: {load_path} ===')
            cache = _load_cache_with_csv_fallback(
                load_path, csv_for_fallback if csv_for_fallback else csv_path)
            cache['pca_latents'] = None  # cache doesn't store the all-star pool
            return cache

        if args.model_path is None:
            parser.error(f'--model_path is required to extract {label} latents '
                         f'(no --load_latents_{label}).')
        print(f'\n=== Extracting {label} latents from {h5_path} ===')
        (lv, css_lat, css_tic,
         a, b, be, mg_, mge_, mp_, gid, tid, sec, pca_pool) = load_data_fresh(
            model_path=args.model_path,
            h5_path=h5_path,
            csv_path=csv_path,
            hidden_size=args.hidden_size,
            direction=args.direction,
            mode=args.mode,
            use_flow=args.use_flow,
            max_length=args.max_length,
            batch_size=32,
            pooling_mode=args.pooling_mode,
            max_samples=args.max_samples,
            stratify_by_age=args.stratify_by_age,
            n_age_bins=args.n_age_bins,
            bprp0_min=args.bprp0_min,
            bprp0_max=args.bprp0_max,
            use_metadata=args.use_metadata,
            use_conv_channels=args.use_conv_channels,
            conv_config=conv_config,
            require_prot=args.require_prot,
            device=device,
            pca_h5_paths=None,  # combined mode aggregates pca pools after both sides extracted
            use_mg=args.use_mg,
            compute_cross_sector=(args.star_aggregation == 'cross_sector'),
            age_loader_fn=age_loader_fn,
        )
        if save_path:
            save_latents_cache(
                save_path, lv, a, b, be, gid, tid, sec, css_lat, css_tic,
                mg=mg_, mg_err=mge_, mem_prob=mp_,
            )
        return dict(
            latent_vectors=lv, ages=a, bprp0=b, bprp0_err=be,
            mg=mg_, mg_err=mge_, mem_prob=mp_,
            gaia_ids=gid, tic_ids=tid, sectors=sec,
            cross_sector_latents=css_lat, cross_sector_tic_ids=css_tic,
            pca_latents=pca_pool,
        )

    if args.combined_kfold:
        if not args.host_h5_path:
            parser.error('--combined_kfold requires --host_h5_path.')
        if not args.load_latents_pretrain and not args.h5_path:
            parser.error('--combined_kfold requires --h5_path for non-host pretrain stars '
                         '(or --load_latents_pretrain).')
        host_age_csv = args.host_age_csv or args.age_csv
        host_loader  = (lambda h5p, csvp, sample_indices=None:
                        load_host_ages(h5p, csvp, args.host_metadata_csv, sample_indices))

        c_p = _load_or_extract_one_source(
            'pretrain', args.h5_path, args.age_csv,
            args.save_latents_pretrain, args.load_latents_pretrain,
            age_loader_fn=None,  # default load_ages
            csv_for_fallback=args.age_csv,
        )
        c_h = _load_or_extract_one_source(
            'hosts', args.host_h5_path, host_age_csv,
            args.save_latents_hosts, args.load_latents_hosts,
            age_loader_fn=host_loader,
            csv_for_fallback=host_age_csv,
        )

        # Concatenate per-sector arrays with a source tag
        n_p, n_h = len(c_p['ages']), len(c_h['ages'])
        source = np.concatenate([np.zeros(n_p, dtype=int), np.ones(n_h, dtype=int)])
        latent_vectors = np.concatenate([c_p['latent_vectors'], c_h['latent_vectors']], axis=0)
        ages           = np.concatenate([c_p['ages'],           c_h['ages']])
        bprp0          = np.concatenate([c_p['bprp0'],          c_h['bprp0']])
        bprp0_err      = np.concatenate([c_p['bprp0_err'],      c_h['bprp0_err']])
        mg             = np.concatenate([c_p['mg'],             c_h['mg']])
        mg_err         = np.concatenate([c_p['mg_err'],         c_h['mg_err']])
        mem_prob       = np.concatenate([c_p['mem_prob'],       c_h['mem_prob']])
        gaia_ids       = np.concatenate([np.asarray(c_p['gaia_ids']),
                                         np.asarray(c_h['gaia_ids'])])
        tic_ids        = np.concatenate([c_p['tic_ids'],        c_h['tic_ids']])
        sectors        = np.concatenate([c_p['sectors'],        c_h['sectors']])

        # Cross-sector arrays: only available if both sides provide them
        if (c_p['cross_sector_latents'] is not None
                and c_h['cross_sector_latents'] is not None):
            cached_cross_sector_latents = np.concatenate(
                [c_p['cross_sector_latents'], c_h['cross_sector_latents']], axis=0)
            cached_cross_sector_tic_ids = np.concatenate(
                [c_p['cross_sector_tic_ids'], c_h['cross_sector_tic_ids']])
        else:
            cached_cross_sector_latents = None
            cached_cross_sector_tic_ids = None

        # Global PCA pool: only when BOTH sides came from fresh extraction
        if c_p['pca_latents'] is not None and c_h['pca_latents'] is not None:
            pca_latents = np.concatenate([c_p['pca_latents'], c_h['pca_latents']], axis=0)
            print(f'Combined PCA pool: {len(pca_latents)} all-star latents '
                  f'({len(c_p["pca_latents"])} pretrain + {len(c_h["pca_latents"])} hosts)')
        else:
            pca_latents = None
            print('Combined mode: at least one side loaded from cache → '
                  'falling back to labeled-subset normalization stats.')

        print(f'Combined dataset: {n_p} pretrain rows + {n_h} host rows = {n_p + n_h} total')

    elif args.load_latents:
        print(f'\n=== Loading latents from cache: {args.load_latents} ===')
        c = _load_cache_with_csv_fallback(args.load_latents, args.age_csv)
        latent_vectors = c['latent_vectors']
        ages, bprp0, bprp0_err = c['ages'], c['bprp0'], c['bprp0_err']
        mg, mg_err, mem_prob = c['mg'], c['mg_err'], c['mem_prob']
        gaia_ids, tic_ids, sectors = c['gaia_ids'], c['tic_ids'], c['sectors']
        cached_cross_sector_latents = c['cross_sector_latents']
        cached_cross_sector_tic_ids = c['cross_sector_tic_ids']

        if args.override_ages_from_csv:
            # Re-join ages from --age_csv (e.g. ChronoFlow isochrone ages) so LOCO
            # grouping + target use clean per-cluster ages, not the blended
            # multi-catalog ages baked into the cache. Drop rows not in the CSV.
            adf = pd.read_csv(args.age_csv); adf['GaiaDR3_ID'] = adf['GaiaDR3_ID'].astype(str)
            # metadata.csv has one row per (star, ref); restrict to the subset
            # (e.g. ref==ChronoFlow) BEFORE mapping so multi-ref stars get the
            # correct catalog's age — matching how the gyro baseline filters.
            if args.subset_col and args.subset_val and args.subset_col in adf.columns:
                adf = adf[adf[args.subset_col].isin(args.subset_val)]
            amap = dict(zip(adf['GaiaDR3_ID'], adf['age_Myr'].astype(float)))
            gids = np.asarray(gaia_ids).astype(str)
            new_ages = np.array([amap.get(g, np.nan) for g in gids], dtype=np.float64)
            keep = np.isfinite(new_ages)
            nchg = int(np.sum(np.isfinite(ages) & keep
                              & (np.abs(new_ages - np.asarray(ages, float)) > 1e-3)))
            print(f'  --override_ages_from_csv ({args.age_csv}): {nchg} ages changed, '
                  f'dropping {int((~keep).sum())} rows not in CSV')
            ages = new_ages.astype(np.float32)
            (latent_vectors, ages, bprp0, bprp0_err, mg, mg_err, mem_prob,
             gaia_ids, tic_ids, sectors) = (
                a[keep] for a in (latent_vectors, ages, bprp0, bprp0_err, mg, mg_err,
                                  mem_prob, gaia_ids, tic_ids, sectors))

        # ── Global PCA fit pool (multi-cache) ──────────────────────────────
        # Without this, run_kfold_cv (encoder_type=pca) refits PCA per-fold on
        # X_train — fine for LOCO but problematic for LOSO, where each fold's
        # training set systematically excludes CVZ-touching stars and thus each
        # fold's basis is OOD on the held-out CVZ stars. Concatenating the three
        # cache files (pretrain + hosts + thickdisk) gives a single basis that
        # spans the full encoder output space, used identically by every fold.
        # PCA is unsupervised, so including the labeled stars' latents in the
        # fit pool introduces no target leakage.
        if args.pca_latent_pool:
            print(f'\n=== Loading PCA fit pool ({len(args.pca_latent_pool)} cache(s)) ===')
            pools = []
            for p in args.pca_latent_pool:
                with np.load(p, allow_pickle=True) as data:
                    lv = data['latent_vectors']
                    pools.append(np.asarray(lv))
                    print(f'  {p}: {pools[-1].shape}')
            pca_latents = np.concatenate(pools, axis=0)
            del pools
            print(f'Global PCA pool: {len(pca_latents):,} latents x {pca_latents.shape[1]} dims')
    else:
        if args.model_path is None:
            parser.error('--model_path is required unless --load_latents is set.')
        (latent_vectors, cached_cross_sector_latents, cached_cross_sector_tic_ids,
         ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors,
         pca_latents) = load_data_fresh(
            model_path=args.model_path,
            h5_path=args.h5_path,
            csv_path=args.age_csv,
            hidden_size=args.hidden_size,
            direction=args.direction,
            mode=args.mode,
            use_flow=args.use_flow,
            max_length=args.max_length,
            batch_size=32,
            pooling_mode=args.pooling_mode,
            max_samples=args.max_samples,
            stratify_by_age=args.stratify_by_age,
            n_age_bins=args.n_age_bins,
            bprp0_min=args.bprp0_min,
            bprp0_max=args.bprp0_max,
            use_metadata=args.use_metadata,
            use_conv_channels=args.use_conv_channels,
            conv_config=conv_config,
            require_prot=args.require_prot,
            device=device,
            pca_h5_paths=args.pca_h5_paths if args.pca_h5_paths else None,
            use_mg=args.use_mg,
            compute_cross_sector=(args.star_aggregation == 'cross_sector'),
        )

        if args.save_latents:
            save_latents_cache(
                args.save_latents,
                latent_vectors, ages, bprp0, bprp0_err, gaia_ids, tic_ids, sectors,
                cached_cross_sector_latents, cached_cross_sector_tic_ids,
                mg=mg, mg_err=mg_err, mem_prob=mem_prob,
            )

    # ── Step 1b: optional subset filter ────────────────────────────────────
    if args.subset_col and args.subset_val:
        csv_for_filter = args.subset_csv or args.age_csv
        df_sub = pd.read_csv(csv_for_filter)
        df_sub['GaiaDR3_ID'] = df_sub['GaiaDR3_ID'].astype(str)
        if args.subset_col not in df_sub.columns:
            parser.error(f'--subset_col "{args.subset_col}" not found in {csv_for_filter}. '
                         f'Available columns: {list(df_sub.columns)}')
        allowed_ids = set(
            df_sub.loc[df_sub[args.subset_col].isin(args.subset_val), 'GaiaDR3_ID']
        )
        keep = np.array([g in allowed_ids for g in gaia_ids])
        n_before = len(ages)
        latent_vectors = latent_vectors[keep]
        ages      = ages[keep]
        bprp0     = bprp0[keep]
        bprp0_err = bprp0_err[keep]
        mg        = mg[keep]
        mg_err    = mg_err[keep]
        mem_prob  = mem_prob[keep]
        gaia_ids  = gaia_ids[keep]
        tic_ids   = tic_ids[keep]
        sectors   = sectors[keep]
        if source is not None:
            source = source[keep]
        n_after = len(ages)
        print(f'\nSubset filter: {args.subset_col} in {args.subset_val} '
              f'→ {n_after}/{n_before} per-sector samples '
              f'({len(np.unique(gaia_ids))} unique stars)')
        if n_after == 0:
            raise ValueError('Subset filter removed all samples — check --subset_col / --subset_val / --subset_csv.')

    # ── Step 2: apply star aggregation ─────────────────────────────────────
    agg = args.star_aggregation
    star_level_split = False  # whether run_kfold_cv should split by unique TIC
    kfold_source = None       # populated only in combined_kfold mode

    # kfold_sectors is per-sector and aligned with kfold_* only in 'none' and
    # 'predict_mean' modes. It feeds the per-sector CSV saved by predict_mean
    # below. Other aggregations collapse sectors into a single row per star.
    kfold_sectors = None

    if agg == 'none':
        kfold_latents   = latent_vectors
        kfold_ages      = ages
        kfold_bprp0     = bprp0
        kfold_bprp0_err = bprp0_err
        kfold_mg        = mg
        kfold_mg_err    = mg_err
        kfold_mem_prob  = mem_prob
        kfold_tics      = tic_ids
        kfold_sectors   = sectors
        kfold_source    = source

    elif agg == 'predict_mean':
        kfold_latents   = latent_vectors
        kfold_ages      = ages
        kfold_bprp0     = bprp0
        kfold_bprp0_err = bprp0_err
        kfold_mg        = mg
        kfold_mg_err    = mg_err
        kfold_mem_prob  = mem_prob
        kfold_tics      = tic_ids
        kfold_sectors   = sectors
        kfold_source    = source
        star_level_split = True

    elif agg in ('latent_mean', 'latent_median', 'latent_max', 'latent_mean_std'):
        print(f'\n=== Aggregating latents ({agg}) ===')
        agg_out = aggregate_by_star(
            latent_vectors, ages, bprp0, bprp0_err, mg, mg_err, mem_prob, tic_ids, gaia_ids,
            method=agg, source=source,
        )
        if source is not None:
            (kfold_latents, kfold_ages, kfold_bprp0, kfold_bprp0_err,
             kfold_mg, kfold_mg_err, kfold_mem_prob, kfold_tics, kfold_source) = agg_out
        else:
            (kfold_latents, kfold_ages, kfold_bprp0, kfold_bprp0_err,
             kfold_mg, kfold_mg_err, kfold_mem_prob, kfold_tics) = agg_out

    elif agg == 'cross_sector':
        print('\n=== Cross-sector hidden-state pooling ===')
        kfold_latents = cached_cross_sector_latents
        # Cache stores TIC IDs; bridge to Gaia IDs for consistent star-level keying
        tic_to_gaia   = dict(zip(tic_ids, gaia_ids))
        kfold_tics    = np.array([tic_to_gaia[t] for t in cached_cross_sector_tic_ids])
        gaia_to_age       = {g: ages[gaia_ids == g][0]      for g in np.unique(gaia_ids)}
        gaia_to_bprp0     = {g: bprp0[gaia_ids == g][0]     for g in np.unique(gaia_ids)}
        gaia_to_bprp0_err = {g: bprp0_err[gaia_ids == g][0] for g in np.unique(gaia_ids)}
        gaia_to_mg        = {g: mg[gaia_ids == g][0]        for g in np.unique(gaia_ids)}
        gaia_to_mg_err    = {g: mg_err[gaia_ids == g][0]    for g in np.unique(gaia_ids)}
        gaia_to_mem_prob  = {g: mem_prob[gaia_ids == g][0]  for g in np.unique(gaia_ids)}
        kfold_ages      = np.array([gaia_to_age[g]       for g in kfold_tics])
        kfold_bprp0     = np.array([gaia_to_bprp0[g]     for g in kfold_tics])
        kfold_bprp0_err = np.array([gaia_to_bprp0_err[g] for g in kfold_tics])
        kfold_mg        = np.array([gaia_to_mg[g]        for g in kfold_tics])
        kfold_mg_err    = np.array([gaia_to_mg_err[g]    for g in kfold_tics])
        kfold_mem_prob  = np.array([gaia_to_mem_prob[g]  for g in kfold_tics])
        if source is not None:
            gaia_to_source = {g: source[gaia_ids == g][0] for g in np.unique(gaia_ids)}
            kfold_source   = np.array([gaia_to_source[g] for g in kfold_tics], dtype=int)

    else:
        raise ValueError(f'Unknown star_aggregation: {agg}')

    # ── Step 2b: optional MG-only swap ─────────────────────────────────────
    # Replace (BPRP0, BPRP0_err) with (MG, MG_err) so the flow context becomes
    # (log10_age, MG, log10(MG_err)) without changing model classes. Filter NaNs
    # in mg/mg_err first since hosts can have a few non-finite values.
    if args.use_mg_only:
        mg_ok = np.isfinite(kfold_mg) & np.isfinite(kfold_mg_err) & (kfold_mg_err > 0)
        n_drop = (~mg_ok).sum()
        if n_drop:
            print(f'\n--use_mg_only: dropping {n_drop}/{len(mg_ok)} rows with NaN MG / MG_err.')
            kfold_latents   = kfold_latents[mg_ok]
            kfold_ages      = kfold_ages[mg_ok]
            kfold_bprp0     = kfold_bprp0[mg_ok]
            kfold_bprp0_err = kfold_bprp0_err[mg_ok]
            kfold_mg        = kfold_mg[mg_ok]
            kfold_mg_err    = kfold_mg_err[mg_ok]
            kfold_mem_prob  = kfold_mem_prob[mg_ok]
            kfold_tics      = kfold_tics[mg_ok]
            if kfold_sectors is not None:
                kfold_sectors = kfold_sectors[mg_ok]
            if kfold_source is not None:
                kfold_source = kfold_source[mg_ok]
        print(f'--use_mg_only: swapping (BPRP0, BPRP0_err) ← (MG, MG_err) '
              f'[{len(kfold_mg)} samples].')
        kfold_bprp0     = kfold_mg
        kfold_bprp0_err = kfold_mg_err

    # ── Step 2c: LOSO per-star sector sets ─────────────────────────────────
    # latent_max collapses all sectors of a star into one row, so the sector
    # info is gone by run_kfold_cv. Rebuild each star's sector-set here, aligned
    # to the aggregated rows. aggregate_by_star keys on np.unique(gaia_ids), so
    # kfold_tics (for latent_max) == the unique Gaia IDs in the same order.
    loso_star_sectors = None
    if args.loso:
        gid_to_secs = {}
        for g, s in zip(gaia_ids, sectors):
            gid_to_secs.setdefault(g, set()).add(int(s))
        missing = [g for g in kfold_tics if g not in gid_to_secs]
        if missing:
            raise ValueError(f'--loso: {len(missing)} aggregated stars have no sector record '
                             '(kfold_tics not aligned to gaia_ids?).')
        loso_star_sectors = [frozenset(gid_to_secs[g]) for g in kfold_tics]
        n_usec = len({s for ss in loso_star_sectors for s in ss})
        print(f'\n--loso: built per-star sector sets for {len(loso_star_sectors)} stars '
              f'across {n_usec} unique sectors.')

    # ── Step 2d: global PCA cache (compute-once / load-once) ───────────────
    # When --pca_cache is set, every run uses bit-identical X normalization +
    # PCA basis: the first run computes from --pca_latent_pool and saves; later
    # runs (sweep at other dims, LOSO) just load and truncate. Eliminates the
    # per-fold PCA refit AND the numerical noise of refitting the global basis
    # across runs.
    prefit_pca_bundle = None
    if args.pca_cache:
        import os
        if os.path.exists(args.pca_cache):
            print(f'\n=== Loading global PCA cache: {args.pca_cache} ===')
            pca_obj_c, X_mean_c, X_std_c = load_global_pca_artifact(args.pca_cache, args.pca_dim)
            prefit_pca_bundle = (pca_obj_c, X_mean_c, X_std_c)
            pca_latents = None  # bundle supersedes the inline fit
        else:
            if pca_latents is None:
                parser.error('--pca_cache file does not exist and --pca_latent_pool was not '
                             'provided; cannot compute the cache.')
            print(f'\n=== Computing global PCA cache (max_dim={args.pca_cache_max_dim}) ===')
            pca_full, Xm, Xs = fit_global_pca_bundle(pca_latents, args.pca_cache_max_dim)
            save_global_pca_artifact(args.pca_cache, pca_full, Xm, Xs)
            del pca_full
            # Reload at the requested dim so the run uses the same code path as
            # subsequent cache-hit runs.
            pca_obj_c, X_mean_c, X_std_c = load_global_pca_artifact(args.pca_cache, args.pca_dim)
            prefit_pca_bundle = (pca_obj_c, X_mean_c, X_std_c)
            pca_latents = None

    # ── Step 3: k-fold cross-validation ────────────────────────────────────
    print('\n=== Running K-Fold Cross-Validation ===')
    predictions, pred_stats, true_ages, result_tics, fold_assignments, fold_losses, _ = run_kfold_cv(
        latent_vectors=kfold_latents,
        ages=kfold_ages,
        bprp0=kfold_bprp0,
        bprp0_err=kfold_bprp0_err,
        mg=kfold_mg,
        mem_prob=kfold_mem_prob,
        tic_ids=kfold_tics,
        n_folds=args.n_folds,
        pca_dim=args.pca_dim,
        pca_latents=pca_latents,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        save_models_dir=output_dir,
        star_level_split=star_level_split,
        flow_transforms=args.flow_transforms,
        flow_hidden_features=args.flow_hidden_dims,
        loga_grid_size=args.loga_grid_size,
        encoder_type=args.encoder_type,
        mlp_encoder_hidden=args.mlp_encoder_hidden,
        aux_loss_weight=args.aux_loss_weight,
        use_mg=args.use_mg,
        lr_decay_rate=args.lr_decay_rate,
        dropout=args.dropout,
        variance_reg_weight=args.variance_reg_weight,
        training_stages=args.training_stages,
        encoder_pretrain_epochs=args.encoder_pretrain_epochs,
        joint_finetune_epochs=args.joint_finetune_epochs,
        finetune_encoder_lr_mult=args.finetune_encoder_lr_mult,
        finetune_flow_lr_mult=args.finetune_flow_lr_mult,
        train_full=args.train_full,
        source=kfold_source,
        sectors=kfold_sectors,
        sector_level_split=args.sector_level_split,
        sector_split_drop_star_overlap=(not args.sector_split_keep_star_overlap),
        balance_sector_age=args.balance_sector_age,
        n_balance_age_bins=args.n_balance_age_bins,
        adv_sector_weight=args.adv_sector_weight,
        adv_hidden=args.adv_hidden,
        loocv_age=args.loocv_age,
        loso=args.loso,
        loso_star_sectors=loso_star_sectors,
        loso_strict_train=(not args.loso_relaxed_train),
        prefit_pca_bundle=prefit_pca_bundle,
    )

    # ── Step 4: for predict_mean, save per-sector CSV then average per star ─
    if agg == 'predict_mean':
        # In combined mode, drop non-host rows here so the per-sector CSV
        # and per-star averages reflect the validated set only.
        if args.combined_kfold and kfold_source is not None:
            ps_mask         = (fold_assignments >= 0)
            result_tics     = result_tics[ps_mask]
            sectors_ps      = kfold_sectors[ps_mask]
            true_ages       = true_ages[ps_mask]
            predictions     = predictions[ps_mask]
            kfold_bprp0     = kfold_bprp0[ps_mask]
            kfold_bprp0_err = kfold_bprp0_err[ps_mask]
            kfold_mg        = kfold_mg[ps_mask]
            kfold_mg_err    = kfold_mg_err[ps_mask]
            kfold_mem_prob  = kfold_mem_prob[ps_mask]
            fold_assignments = fold_assignments[ps_mask]
            if pred_stats is not None:
                pred_stats = {k: v[ps_mask] for k, v in pred_stats.items()}
        else:
            sectors_ps = kfold_sectors

        # Save per-sector predictions first — useful for uncertainty estimation
        # (spread of per-sector predictions = observational scatter per star).
        # Column name reflects what kfold_bprp0 actually carries, since
        # --use_mg_only swaps MG into the BPRP0 slot.
        photom_col = 'mg' if args.use_mg_only else 'bprp0'
        pd.DataFrame({
            'TIC_ID': result_tics,
            'sector': sectors_ps,
            'true_age_myr': true_ages,
            'predicted_age_myr': predictions,
            photom_col: kfold_bprp0,
            'fold': fold_assignments,
        }).to_csv(output_dir / 'kfold_predictions_per_sector.csv', index=False)
        print(f'Saved per-sector predictions to {output_dir / "kfold_predictions_per_sector.csv"}')

        print('\nAveraging per-sector predictions per star...')
        unique_tics = np.unique(result_tics)
        per_sector_tics = result_tics
        predictions     = np.array([predictions[per_sector_tics == t].mean()       for t in unique_tics])
        true_ages       = np.array([true_ages[per_sector_tics == t][0]             for t in unique_tics])
        kfold_bprp0     = np.array([kfold_bprp0[per_sector_tics == t][0]           for t in unique_tics])
        kfold_bprp0_err = np.array([kfold_bprp0_err[per_sector_tics == t][0]       for t in unique_tics])
        kfold_mg        = np.array([kfold_mg[per_sector_tics == t][0]              for t in unique_tics])
        kfold_mg_err    = np.array([kfold_mg_err[per_sector_tics == t][0]          for t in unique_tics])
        kfold_mem_prob  = np.array([kfold_mem_prob[per_sector_tics == t][0]        for t in unique_tics])
        fold_assignments = np.array([fold_assignments[per_sector_tics == t][0]     for t in unique_tics])
        if pred_stats is not None:
            pred_stats = {
                k: np.array([v[per_sector_tics == t].mean() for t in unique_tics])
                for k, v in pred_stats.items()
            }
        result_tics     = unique_tics
        print(f'Averaged to {len(unique_tics)} unique stars')

    # In combined_kfold mode, only host rows have a real fold assignment;
    # restrict plotting/metrics to those rows.
    if args.combined_kfold and kfold_source is not None:
        # 'predict_mean' aggregation may have collapsed source array; rebuild
        # by filtering on fold_assignments != -1 (only hosts ever got a fold).
        host_mask = (fold_assignments >= 0)
        n_before = len(predictions)
        predictions = predictions[host_mask]
        true_ages   = true_ages[host_mask]
        kfold_bprp0 = kfold_bprp0[host_mask]
        result_tics = result_tics[host_mask]
        fold_assignments = fold_assignments[host_mask]
        if pred_stats is not None:
            pred_stats = {k: v[host_mask] for k, v in pred_stats.items()}
        print(f'\nCombined mode: filtered to {len(predictions)} host rows '
              f'(from {n_before} total).')

    # ── Step 5: results ─────────────────────────────────────────────────────
    plot_kfold_results(
        predictions, true_ages, kfold_bprp0, result_tics, fold_assignments, output_dir,
        pred_stats=pred_stats,
    )

    config = vars(args)
    config.update({
        'latent_dim':    int(kfold_latents.shape[1]),
        'n_samples':     int(len(kfold_ages)),
        'fold_losses':   [float(l) for l in fold_losses],
        'star_level_split': star_level_split,
    })
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print('\nDone!')


if __name__ == '__main__':
    main()
