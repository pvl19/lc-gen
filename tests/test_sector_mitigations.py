"""Tests for the sector-confound mitigations in kfold_age_inference.py.

  Option 1  sector-disjoint CV        (--sector_level_split)
  Option 2  (sector x age) balancing  (--balance_sector_age)
  Option 3  GRL sector adversary      (--adv_sector_weight)

Unit tests cover GradReverse and the adversary head; integration tests drive
run_kfold_cv end-to-end on tiny synthetic data so the full wiring (7-tuple
dataset, weighted sampler, λ ramp, NaN handling) is exercised.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kfold_age_inference import GradReverse, AgePredictorMLP, run_kfold_cv  # noqa: E402


# ----------------------------- unit: GRL ----------------------------------- #
def test_grad_reverse_identity_forward_negated_backward():
    x = torch.randn(5, 3, requires_grad=True)
    y = GradReverse.apply(x, 2.5)
    assert torch.allclose(y, x)                       # identity forward
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -2.5))  # gradient × (−λ)


# ------------------------- unit: adversary head ---------------------------- #
def _dummy_batch(n=24, D=16, n_sectors=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        x=torch.randn(n, D, generator=g), log_age=torch.rand(n, generator=g) * 3,
        bprp0=torch.rand(n, generator=g), berr=torch.rand(n, generator=g),
        mg=torch.rand(n, generator=g), mem=torch.full((n,), 0.9),
        sec=torch.randint(0, n_sectors, (n,), generator=g))


def test_adversary_builds_and_feeds_gradient_to_encoder():
    m = AgePredictorMLP(input_dim=16, bottleneck_dim=4, mlp_hidden=[8],
                        flow_transforms=2, flow_hidden_features=[8],
                        n_sectors=5, adv_hidden=8)
    assert m.sector_adversary is not None
    b = _dummy_batch()
    loss = m(b['x'], b['log_age'], b['bprp0'], b['berr'], b['mg'], b['mem'],
             loss_mode='full', sector_idx=b['sec'], adv_lambda=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
    adv_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.sector_adversary.parameters())
    assert enc_grad and adv_grad   # GRL feeds both the encoder and the adversary


def test_no_adversary_when_n_sectors_zero():
    m = AgePredictorMLP(input_dim=16, bottleneck_dim=4, mlp_hidden=[8],
                        flow_transforms=2, flow_hidden_features=[8], n_sectors=0)
    assert m.sector_adversary is None
    b = _dummy_batch()
    # sector args are accepted but ignored — loss stays finite.
    loss = m(b['x'], b['log_age'], b['bprp0'], b['berr'], b['mg'], b['mem'],
             loss_mode='full', sector_idx=b['sec'], adv_lambda=1.0)
    assert torch.isfinite(loss)


# --------------------- integration: run_kfold_cv --------------------------- #
def _synth(n_stars=40, n_sectors=6, D=16, seed=0):
    rng = np.random.default_rng(seed)
    tic, sec = [], []
    for s in range(n_stars):
        k = int(rng.integers(1, 4))
        for sv in rng.choice(n_sectors, size=min(k, n_sectors), replace=False):
            tic.append(s); sec.append(int(sv))
    N = len(tic)
    return dict(
        latent_vectors=rng.standard_normal((N, D)).astype(np.float32),
        ages=(10 ** rng.uniform(1.0, 3.3, N)).astype(np.float32),
        bprp0=rng.uniform(0.5, 2.5, N).astype(np.float32),
        bprp0_err=rng.uniform(0.01, 0.1, N).astype(np.float32),
        mg=rng.uniform(0.0, 8.0, N).astype(np.float32),
        mem_prob=np.full(N, 0.9, np.float32),
        tic_ids=np.array(tic), sectors=np.array(sec))


def _run(extra, d=None, save_models_dir=None):
    d = d or _synth()
    return run_kfold_cv(
        d['latent_vectors'], d['ages'], d['bprp0'], d['bprp0_err'], d['mg'],
        d['mem_prob'], d['tic_ids'], n_folds=3, pca_dim=4, lr=1e-3,
        weight_decay=1e-4, n_epochs=2, batch_size=16, device=torch.device('cpu'),
        seed=0, encoder_type='mlp', mlp_encoder_hidden=[8], flow_transforms=2,
        flow_hidden_features=[8], training_stages='joint', sectors=d['sectors'],
        save_models_dir=save_models_dir, **extra)


def test_sector_disjoint_cv_runs_and_drops_overlap():
    preds, stats, ages, tics, folds, losses, _ = _run({'sector_level_split': True})
    assert len(preds) == len(ages)
    assert np.isfinite(preds).any()                   # some rows predicted
    # star-overlap drop should leave at least one unpredicted (NaN) row
    assert np.isnan(preds).any()


def test_sector_disjoint_with_save_survives_skipped_fold(tmp_path):
    """A held-out sector whose stars all appear in training sectors empties a fold;
    the model/curve save must index by the actual fold count, not n_folds."""
    rng = np.random.default_rng(1)
    tic, sec = [], []
    # Stars 0-29 each span sectors {0,1,2} → in 3-fold sector-disjoint CV, every
    # held-out sector's stars also appear in a training sector → folds get emptied.
    for s in range(30):
        for sv in (0, 1, 2):
            tic.append(s); sec.append(sv)
    N = len(tic)
    d = dict(
        latent_vectors=rng.standard_normal((N, 16)).astype(np.float32),
        ages=(10 ** rng.uniform(1.0, 3.3, N)).astype(np.float32),
        bprp0=rng.uniform(0.5, 2.5, N).astype(np.float32),
        bprp0_err=rng.uniform(0.01, 0.1, N).astype(np.float32),
        mg=rng.uniform(0.0, 8.0, N).astype(np.float32),
        mem_prob=np.full(N, 0.9, np.float32),
        tic_ids=np.array(tic), sectors=np.array(sec))
    # Must not raise (this is the IndexError regression).
    _run({'sector_level_split': True}, d=d, save_models_dir=str(tmp_path))
    assert (tmp_path / 'kfold_models.pt').exists()


def test_balance_sector_age_runs():
    preds, *_ = _run({'balance_sector_age': True, 'n_balance_age_bins': 5})
    assert np.isfinite(preds).all()                   # normal split predicts every row


def test_adversary_runs_with_mlp():
    preds, *_ = _run({'adv_sector_weight': 1.0, 'adv_hidden': 8})
    assert np.isfinite(preds).all()


def test_adversary_rejects_pca_encoder():
    d = _synth()
    with pytest.raises(ValueError):
        run_kfold_cv(d['latent_vectors'], d['ages'], d['bprp0'], d['bprp0_err'],
                     d['mg'], d['mem_prob'], d['tic_ids'], n_folds=3, pca_dim=4,
                     n_epochs=1, batch_size=16, device=torch.device('cpu'),
                     encoder_type='pca', sectors=d['sectors'], adv_sector_weight=1.0)


def test_loocv_age_folds_grouping():
    """Shared LOCO grouping: young bundled in 5s, big clusters individual, sparse bundled."""
    from kfold_age_inference import loocv_age_folds
    ages = [float(a) for a in range(1, 21) for _ in range(3)]   # 20 young ages, 3 stars each
    ages += [100.0] * 150                                       # big cluster -> individual
    ages += [500.0] * 5                                         # sparse mid (<1000) -> bundle
    ages += [2000.0] * 4                                        # sparse old (>=1000) -> bundle
    ages = np.array(ages)
    fos, nf = loocv_age_folds(ages, young_n=20, young_size=5, indiv_min=100, sparse_split=1000.)
    assert nf == 7                                              # 4 young + 1 big + mid + old
    assert len(np.unique(fos[np.isin(ages, [1., 2., 3., 4., 5.])])) == 1   # youngest 5 share a fold
    assert len(np.unique(fos[ages == 100.])) == 1              # big cluster is its own fold
    assert fos[ages == 500.][0] != fos[ages == 2000.][0]       # mid vs old in different folds
    assert (fos >= 0).all() and len(np.unique(fos)) == 7


def test_loocv_age_runs_end_to_end():
    """run_kfold_cv with loocv_age completes and predicts every star (grouped folds)."""
    rng = np.random.default_rng(3)
    # 22 ages: 20 young (3 stars) + a big cluster (40) + a sparse-old (4)
    ages = np.array([float(a) for a in range(1, 21) for _ in range(3)]
                    + [100.0] * 40 + [2000.0] * 4, dtype=np.float32)
    N = len(ages)
    d = dict(
        latent_vectors=rng.standard_normal((N, 16)).astype(np.float32),
        ages=ages, bprp0=rng.uniform(.5, 2.5, N).astype(np.float32),
        bprp0_err=rng.uniform(.01, .1, N).astype(np.float32),
        mg=rng.uniform(0, 8, N).astype(np.float32), mem_prob=np.full(N, .9, np.float32),
        tic_ids=np.arange(N), sectors=rng.integers(0, 6, N))
    preds, stats, a, tics, folds, losses, _ = _run({'loocv_age': True}, d=d)
    assert np.isfinite(preds).all()              # every star held out once and predicted


def test_sector_options_require_sectors():
    d = _synth()
    with pytest.raises(ValueError):
        run_kfold_cv(d['latent_vectors'], d['ages'], d['bprp0'], d['bprp0_err'],
                     d['mg'], d['mem_prob'], d['tic_ids'], n_folds=3, pca_dim=4,
                     n_epochs=1, batch_size=16, device=torch.device('cpu'),
                     encoder_type='mlp', mlp_encoder_hidden=[8], flow_transforms=2,
                     flow_hidden_features=[8], sectors=None, balance_sector_age=True)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
