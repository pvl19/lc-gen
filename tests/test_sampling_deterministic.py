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
        input_length=32,
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


def make_inputs(batch=2, seq_len=32, device='cpu'):
    # channels: [masked_flux, flux_err, mask]
    x = torch.zeros(batch, seq_len, 3, device=device)
    # set a small measurement error
    x[:, :, 1] = 0.1
    # mask: True = masked; start with all masked
    mask = torch.ones(batch, seq_len, dtype=torch.bool, device=device)
    # make timestamps simple range
    t = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1).float()
    # simple arbitrary target
    target = torch.randn(batch, seq_len, device=device)
    return x, t, target, mask


def test_sampling_deterministic_same_seed():
    device = 'cpu'
    model = make_model(device=device)
    x, t, target, mask = make_inputs(batch=3, seq_len=24, device=device)

    # Run twice with same seed and assert sampled traces identical
    out1 = model.forward(x, t=t, target=target, mask=mask, seed=12345)
    out2 = model.forward(x, t=t, target=target, mask=mask, seed=12345)

    s1 = out1.get('sampled')
    s2 = out2.get('sampled')

    assert s1 is not None and s2 is not None, "Sampled outputs missing"
    assert torch.allclose(s1, s2), "Samples differ for identical seed"
