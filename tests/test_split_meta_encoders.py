"""Unit tests for split metadata encoders + adversarial sector head.

Run: python tests/test_split_meta_encoders.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from lcgen.models.simple_min_gru import BiDirectionalMinGRU, grad_reverse
from lcgen.utils.loss import bounded_horizon_future_nll


def _inputs(B=2, L=14):
    x = torch.randn(B, L, 2)
    t = torch.linspace(0, 1, L).unsqueeze(0).expand(B, L).unsqueeze(-1).contiguous()
    mask = torch.ones(B, L)
    metadata = torch.randn(B, 13)
    return x, t, mask, metadata


def test_grad_reverse():
    x = torch.randn(3, 4, requires_grad=True)
    y = grad_reverse(x, 2.0)
    assert torch.allclose(y, x), 'GRL must be identity on the forward pass'
    y.sum().backward()
    # d(sum)/dx = 1, reversed and scaled by lambda=2 -> -2
    assert torch.allclose(x.grad, torch.full_like(x.grad, -2.0))
    print('ok: gradient reversal layer — identity forward, gradient x(-lambda) backward')


def test_split_builds_two_encoders():
    m = BiDirectionalMinGRU(hidden_size=8, num_meta_features=13, use_flow=False,
                            split_meta_encoders=True, instr_emb_dim=16)
    assert m.meta_encoder.input_dim == 9, 'astro encoder should take 9 fields'
    assert m.instr_meta_encoder.input_dim == 3, 'instrumental encoder should take 3 fields'
    assert m.instr_emb_dim == 16
    assert len(m.astro_meta_idx) == 9 and len(m.instr_meta_idx) == 3
    print('ok: split builds a 9-field astro encoder + 3-field instrumental encoder')


def test_instrumental_metadata_not_in_hidden_states():
    """THE core property: changing instrumental metadata must NOT change the
    RNN hidden states (it only conditions the head)."""
    torch.manual_seed(0)
    m = BiDirectionalMinGRU(hidden_size=8, num_meta_features=13, use_flow=False,
                            split_meta_encoders=True, instr_emb_dim=16)
    m.eval()
    x, t, mask, meta_a = _inputs()
    meta_b = meta_a.clone()
    meta_b[:, m.instr_meta_idx] = torch.randn(2, 3)  # change ONLY sector/camera/ccd

    h_a = m(x, t, mask=mask, metadata=meta_a, return_states=True)['h_fwd_tensor']
    h_b = m(x, t, mask=mask, metadata=meta_b, return_states=True)['h_fwd_tensor']
    assert torch.allclose(h_a, h_b, atol=1e-6), \
        'instrumental metadata leaked into the RNN hidden states'

    # And changing an ASTRO field DOES change the hidden states (sanity).
    meta_c = meta_a.clone()
    meta_c[:, m.astro_meta_idx[0]] += 3.0
    h_c = m(x, t, mask=mask, metadata=meta_c, return_states=True)['h_fwd_tensor']
    assert not torch.allclose(h_a, h_c, atol=1e-6), 'astro metadata should affect hidden states'
    print('ok: instrumental metadata stays OUT of hidden states; astro metadata stays in')


def test_instrumental_metadata_reaches_the_head():
    """Instrumental metadata must still influence the model's output (via the head)."""
    torch.manual_seed(1)
    m = BiDirectionalMinGRU(hidden_size=8, num_meta_features=13, use_flow=False,
                            split_meta_encoders=True, instr_emb_dim=16)
    m.eval()
    x, t, mask, meta_a = _inputs()
    meta_b = meta_a.clone()
    meta_b[:, m.instr_meta_idx] = torch.randn(2, 3)

    emb_a = m(x, t, mask=mask, metadata=meta_a, return_states=True)['instr_emb']
    emb_b = m(x, t, mask=mask, metadata=meta_b, return_states=True)['instr_emb']
    assert emb_a.shape == (2, 16)
    assert not torch.allclose(emb_a, emb_b, atol=1e-6), \
        'instrumental embedding did not respond to instrumental metadata'
    print('ok: instrumental metadata produces a (B,16) embedding that feeds the head')


def test_adversary_train_vs_eval():
    torch.manual_seed(2)
    m = BiDirectionalMinGRU(hidden_size=8, num_meta_features=13, use_flow=False,
                            adversarial_sector_head=True, num_sectors=20)
    x, t, mask, metadata = _inputs()

    m.train()
    out = m(x, t, mask=mask, metadata=metadata, return_states=True, adv_lambda=1.0)
    assert 'sector_logits' in out and out['sector_logits'].shape == (2, 20)

    m.eval()
    out_e = m(x, t, mask=mask, metadata=metadata, return_states=True)
    assert 'sector_logits' not in out_e, 'adversary must not run at eval time'
    print('ok: adversary emits (B,num_sectors) logits in train mode, silent at eval')


def test_loss_with_instr_emb_runs():
    torch.manual_seed(3)
    m = BiDirectionalMinGRU(hidden_size=8, num_meta_features=13, use_flow=True,
                            split_meta_encoders=True, instr_emb_dim=16)
    m.train()
    x, t, mask, metadata = _inputs(B=2, L=40)
    out = m(x, t, mask=mask, metadata=metadata, return_states=True)
    loss, _, _ = bounded_horizon_future_nll(
        out['h_fwd_tensor'], out['h_bwd_tensor'], out['t_enc'], m,
        x[..., 0], x[..., 1], mask=mask, K=8, k_spacing='log',
        times=t.squeeze(-1), instr_emb=out['instr_emb'])
    assert torch.isfinite(loss), 'loss with instr_emb must be finite'
    loss.backward()
    print('ok: bounded_horizon_future_nll runs + backprops with instr_emb in the flow context')


if __name__ == '__main__':
    test_grad_reverse()
    test_split_builds_two_encoders()
    test_instrumental_metadata_not_in_hidden_states()
    test_instrumental_metadata_reaches_the_head()
    test_adversary_train_vs_eval()
    test_loss_with_instr_emb_runs()
    print('\nAll split-meta-encoder tests passed.')
