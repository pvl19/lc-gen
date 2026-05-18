"""Unit tests for hard recurrence gating in the minGRU scans.

Gating contract: at positions where mask == 0 the minGRU update gate is forced
to 0, so h_t = h_{t-1} — the masked/padded step carries the state through
unchanged, contributes nothing to the recurrence, and receives zero gradient.

Run: python tests/test_recurrence_gating.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from lcgen.models.simple_min_gru import minGRUCell, BiDirectionalMinGRU


def test_step_parallel_carries_state_across_masked_block():
    torch.manual_seed(0)
    cell = minGRUCell(input_size=5, hidden_size=4)
    B, T, Hin = 2, 12, 5
    x = torch.randn(B, T, Hin)
    h0 = torch.zeros(B, 1, 4)

    # Gate an interior block: positions 4,5,6,7.
    mask = torch.ones(B, T)
    mask[:, 4:8] = 0.0

    h = cell.step_parallel(x, h0, mask=mask)  # (B, T, 4): state AFTER step t

    # Across the gated block the state is constant and equals the last
    # pre-block state (h[3]).
    for t in (4, 5, 6, 7):
        assert torch.allclose(h[:, t], h[:, 3], atol=1e-5), f'h[{t}] != h[3]'
    # Positions before the block are untouched relative to the ungated run.
    h_full = cell.step_parallel(x, h0, mask=None)
    assert torch.allclose(h[:, :4], h_full[:, :4], atol=1e-5)
    print('ok: step_parallel carries state across a masked block')


def test_gating_equals_deletion():
    """Gating a block must equal physically deleting those rows for every
    surviving position (the encoder behaves as if the block did not exist)."""
    torch.manual_seed(1)
    cell = minGRUCell(input_size=3, hidden_size=6)
    B, T, Hin = 1, 10, 3
    x = torch.randn(B, T, Hin)
    h0 = torch.zeros(B, 1, 6)

    mask = torch.ones(B, T)
    mask[:, 3:6] = 0.0  # gate positions 3,4,5
    h_gated = cell.step_parallel(x, h0, mask=mask)

    # Physically delete rows 3,4,5.
    keep_idx = [0, 1, 2, 6, 7, 8, 9]
    x_del = x[:, keep_idx, :]
    h_del = cell.step_parallel(x_del, h0, mask=None)

    # Surviving positions must match between the two runs.
    for gated_t, del_t in zip(keep_idx, range(len(keep_idx))):
        assert torch.allclose(h_gated[:, gated_t], h_del[:, del_t], atol=1e-5), \
            f'gated[{gated_t}] != deleted[{del_t}]'
    print('ok: gating a block == deleting it (for surviving positions)')


def test_gated_positions_get_zero_gradient():
    torch.manual_seed(2)
    cell = minGRUCell(input_size=4, hidden_size=5)
    B, T, Hin = 2, 9, 4
    x = torch.randn(B, T, Hin, requires_grad=True)
    h0 = torch.zeros(B, 1, 5)

    mask = torch.ones(B, T)
    mask[:, 2:5] = 0.0
    h = cell.step_parallel(x, h0, mask=mask)
    h.sum().backward()

    grad_gated = x.grad[:, 2:5]
    assert torch.count_nonzero(grad_gated) == 0, 'gated inputs received nonzero gradient'
    # Sanity: ungated positions DO receive gradient.
    assert torch.count_nonzero(x.grad[:, :2]) > 0
    print('ok: gated input positions receive exactly zero gradient')


def test_step_sequential_gating():
    torch.manual_seed(3)
    cell = minGRUCell(input_size=3, hidden_size=4)
    B = 2
    h_prev = torch.randn(B, 4)
    x_t = torch.randn(B, 3)

    # mask_t == 0 for row 0, == 1 for row 1.
    mask_t = torch.tensor([0.0, 1.0])
    h = cell.step(x_t, h_prev, mask_t=mask_t)
    h_ungated = cell.step(x_t, h_prev, mask_t=None)

    assert torch.allclose(h[0], h_prev[0], atol=1e-6), 'gated row did not carry state'
    assert torch.allclose(h[1], h_ungated[1], atol=1e-6), 'ungated row changed'
    print('ok: sequential step gating carries state for masked rows')


def test_bidirectional_forward_with_mask():
    """End-to-end: BiDirectionalMinGRU.forward runs with a mask and the
    forward hidden state is constant across a masked interior block."""
    torch.manual_seed(4)
    model = BiDirectionalMinGRU(hidden_size=8, direction='bi', mode='parallel',
                                use_flow=False, num_meta_features=0)
    model.eval()
    B, L = 2, 16
    x = torch.randn(B, L, 2)
    t = torch.linspace(0, 1, L).unsqueeze(0).expand(B, L).unsqueeze(-1).contiguous()
    mask = torch.ones(B, L)
    mask[:, 6:11] = 0.0

    out = model(x, t, mask=mask, return_states=True)
    h_fwd = out['h_fwd_tensor']  # (B, L, H): state BEFORE processing t
    assert h_fwd is not None and h_fwd.shape == (B, L, 8)
    # h_fwd_tensor[t] = state before t. For gated steps 7..11 the "before"
    # state equals the state before the first gated step (7).
    for t_idx in (8, 9, 10, 11):
        assert torch.allclose(h_fwd[:, t_idx], h_fwd[:, 7], atol=1e-5), \
            f'h_fwd[{t_idx}] != h_fwd[7]'
    print('ok: BiDirectionalMinGRU.forward gates the recurrence on `mask`')


if __name__ == '__main__':
    test_step_parallel_carries_state_across_masked_block()
    test_gating_equals_deletion()
    test_gated_positions_get_zero_gradient()
    test_step_sequential_gating()
    test_bidirectional_forward_with_mask()
    print('\nAll recurrence-gating tests passed.')
