"""Sanity test: the log-domain parallel scan must compute the same recurrence
as a straightforward sequential implementation using the same activation `g`.

Rationale
---------
The parallel scan trades a Python-level `for` loop for two cumulative ops in
log space (`cumsum` + `logcumsumexp`). It implements the recurrence

    h_t = (1 - σ(k_t)) · h_{t-1} + σ(k_t) · g(W_h x_t)

where `g(x) = relu(x) + 0.5` for x>=0, `sigmoid(x)` for x<0 (always positive,
required for the log-domain trick). This test checks that the parallel scan
produces hidden states matching a manually-stepped sequential reference using
the *same* activation. It does NOT compare against the production
`minGRUCell.step`, which deliberately uses `tanh` (a different activation,
chosen because sequential mode does not need positivity).

This test would have caught the `5 - F.softplus(-x)` bug that previously made
parallel hidden states ~30× too large compared to the intended dynamics.

Run from the repo root:
    pytest tests/test_parallel_scan_equivalence.py
or
    python tests/test_parallel_scan_equivalence.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lcgen.models.simple_min_gru import minGRUCell


def _g(x: torch.Tensor) -> torch.Tensor:
    """Reference activation matching the parallel scan's intended dynamics."""
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def _sequential_reference(cell: minGRUCell, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Plain sequential implementation of the recurrence the parallel scan
    is supposed to compute. Uses the same `g` activation (not tanh).

    Args:
        cell: minGRUCell whose W_z, W_h linear layers are used as-is.
        x:    (B, T, in_size) inputs.
        h0:   (B, hidden) initial state.

    Returns:
        h: (B, T, hidden) — h[:, t] is the state AFTER ingesting position t.
    """
    B, T, _ = x.shape
    h_prev = h0
    out = []
    for t in range(T):
        z = torch.sigmoid(cell.W_z(x[:, t]))
        h_tilde = _g(cell.W_h(x[:, t]))
        h_prev = (1.0 - z) * h_prev + z * h_tilde
        out.append(h_prev)
    return torch.stack(out, dim=1)


def test_parallel_scan_matches_sequential_reference():
    torch.manual_seed(0)
    B, T, in_size, hidden = 4, 32, 6, 8
    cell = minGRUCell(in_size, hidden)
    x = torch.randn(B, T, in_size)
    h0 = torch.zeros(B, hidden)

    h_par = cell.step_parallel(x, h0.unsqueeze(1))
    h_seq = _sequential_reference(cell, x, h0)

    abs_diff = (h_par - h_seq).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    seq_mag = h_seq.abs().mean().item()

    print(f'  max abs diff:  {max_diff:.3e}')
    print(f'  mean abs diff: {mean_diff:.3e}')
    print(f'  seq mean |h|:  {seq_mag:.3e}')

    # Tolerance: log-domain math accumulates float32 error proportional to T.
    # For T=32, hidden=8, ~1e-5 atol is comfortable; rtol guards against drift.
    assert torch.allclose(h_par, h_seq, atol=1e-5, rtol=1e-4), (
        f'parallel scan diverges from sequential reference: max_diff={max_diff:.3e} '
        f'(seq mean magnitude={seq_mag:.3e}). This usually means log_g is wrong.'
    )


def test_parallel_scan_long_sequence():
    """Same test but with a longer sequence to exercise compounding error."""
    torch.manual_seed(1)
    B, T, in_size, hidden = 2, 512, 4, 16
    cell = minGRUCell(in_size, hidden)
    x = torch.randn(B, T, in_size)
    h0 = torch.zeros(B, hidden)

    h_par = cell.step_parallel(x, h0.unsqueeze(1))
    h_seq = _sequential_reference(cell, x, h0)

    max_diff = (h_par - h_seq).abs().max().item()
    seq_mag = h_seq.abs().mean().item()
    print(f'  long-T max abs diff: {max_diff:.3e}  (seq mean |h|: {seq_mag:.3e})')

    # Slightly looser tolerance for T=512
    assert torch.allclose(h_par, h_seq, atol=1e-4, rtol=1e-3), (
        f'parallel scan diverges on long sequence: max_diff={max_diff:.3e}'
    )


if __name__ == '__main__':
    print('test_parallel_scan_matches_sequential_reference:')
    test_parallel_scan_matches_sequential_reference()
    print('  PASS\n')
    print('test_parallel_scan_long_sequence:')
    test_parallel_scan_long_sequence()
    print('  PASS')
