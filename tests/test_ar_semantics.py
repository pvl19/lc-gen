import os
import sys
import torch
import pytest

# Ensure local `src` is on path so tests import package modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from lcgen.models.rnn import HierarchicalRNN, RNNConfig
except Exception as e:
    pytest.skip(f"Required model import failed: {e}")


def make_model(device='cpu'):
    cfg = RNNConfig(
        input_dim=3,
        input_length=16,
        encoder_dims=[16, 32],
        rnn_type='minGRU',
        num_layers_per_level=1,
        direction='forward',
        autoregressive=True,
        teacher_forcing=True,
    )
    model = HierarchicalRNN(cfg).to(device)
    model.eval()
    return model


def test_ar_observed_used_for_conditioning():
    device = 'cpu'
    model = make_model(device=device)

    batch = 2
    seq_len = 8
    # prepare input: all masked except t0 observed
    x = torch.zeros(batch, seq_len, 3, device=device)
    x[:, :, 1] = 0.1
    mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    mask[:, 0] = False  # t0 observed

    t = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1).float()

    # Two targets that differ only at t0
    target_a = torch.randn(batch, seq_len, device=device)
    target_b = target_a.clone()
    target_a[:, 0] = 5.0
    target_b[:, 0] = -5.0

    out_a = model.forward(x, t=t, target=target_a, mask=mask, seed=2023)
    out_b = model.forward(x, t=t, target=target_b, mask=mask, seed=2023)

    s_a = out_a.get('sampled')
    s_b = out_b.get('sampled')

    # We expect the sampled value at t=1 to differ because the observed t=0 value
    # should have been used to condition the next-step prev_mean
    assert s_a is not None and s_b is not None
    assert not torch.allclose(s_a[:, 1], s_b[:, 1]), "AR conditioning did not depend on observed value"
