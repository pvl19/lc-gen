import torch
import numpy as np
from src.lcgen.models.rnn import HierarchicalRNN, RNNConfig


def make_dummy_batch(batch_size=2, seq_len=64):
    # Create random flux, small flux_err, and mask
    flux = np.random.randn(batch_size, seq_len).astype(np.float32)
    flux_err = (0.01 * np.ones_like(flux)).astype(np.float32)
    time = np.linspace(0, 1, seq_len).astype(np.float32)[None, :].repeat(batch_size, axis=0)
    # mask: 1 indicates observed, 0 masked
    mask = np.ones_like(flux, dtype=np.float32)
    # target equals original flux
    target = flux.copy()
    # Build input tensor [flux, flux_err, mask]
    inp = np.stack([flux, flux_err, mask], axis=-1)
    return torch.from_numpy(inp), torch.from_numpy(time), torch.from_numpy(target), torch.from_numpy(mask)


def test_autoregressive_sampling_reproducible_and_variable():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 64
    x, t, target, mask = make_dummy_batch(batch_size, seq_len)

    config = RNNConfig(
        input_dim=3,
        input_length=seq_len,
        encoder_dims=[32, 64],
        rnn_type='gru',
        num_layers_per_level=1,
        dropout=0.0,
        min_period=0.001,
        max_period=1.0,
        autoregressive=True,
        teacher_forcing=False,  # evaluate sampling behavior (no teacher forcing)
    )

    model = HierarchicalRNN(config)
    model.eval()

    # Run with same seed twice: sampled outputs should match exactly
    out1 = model(x, t, target=target, mask=mask, seed=123)
    out2 = model(x, t, target=target, mask=mask, seed=123)

    assert isinstance(out1, dict)
    assert 'sampled' in out1
    s1 = out1['sampled']
    s2 = out2['sampled']
    assert s1.shape == (batch_size, seq_len)
    # Exact equality when seed is identical
    assert torch.allclose(s1, s2, atol=0.0)

    # Run with a different seed: outputs should differ (at least one value)
    out3 = model(x, t, target=target, mask=mask, seed=456)
    s3 = out3['sampled']
    # Ensure there's at least one difference across the batch
    assert not torch.allclose(s1, s3, atol=0.0)
