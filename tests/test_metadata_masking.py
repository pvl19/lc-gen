"""Unit tests for DOROTHY-style metadata masking and its model wiring.

Run: python tests/test_metadata_masking.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from lcgen.utils.metadata_masking import DynamicMetadataMasking
from lcgen.models.simple_min_gru import BiDirectionalMinGRU


def test_block_drop_all():
    torch.manual_seed(0)
    masker = DynamicMetadataMasking(p_block=1.0)
    meta = torch.randn(8, 13)
    out, mask_out, block_drop = masker(meta)
    assert block_drop.all(), 'p_block=1 should drop every star'
    assert torch.count_nonzero(out) == 0, 'block-dropped metadata must be all zero'
    assert torch.count_nonzero(mask_out) == 0, 'block-dropped mask must be all zero'
    print('ok: p_block=1.0 drops the whole encoder for every star')


def test_no_masking():
    torch.manual_seed(1)
    masker = DynamicMetadataMasking(p_block=0.0, p_keep_min=1.0, p_keep_max=1.0)
    meta = torch.randn(8, 13)
    out, mask_out, block_drop = masker(meta)
    assert not block_drop.any()
    assert torch.allclose(out, meta), 'no masking should leave metadata untouched'
    assert (mask_out == 1).all(), 'no masking should leave mask all ones'
    print('ok: p_block=0, p_keep=1 is a no-op')


def test_field_masking_keeps_at_least_one():
    torch.manual_seed(2)
    # Aggressive field masking, no block drop.
    masker = DynamicMetadataMasking(p_block=0.0, p_keep_min=0.0, p_keep_max=0.0)
    meta = torch.randn(64, 13)
    out, mask_out, block_drop = masker(meta)
    assert not block_drop.any()
    per_star_kept = mask_out.sum(dim=1)
    assert (per_star_kept >= 1).all(), 'guaranteed-keeper must retain >=1 field per star'
    assert (per_star_kept == 1).all(), 'p_keep=0 should keep exactly the guaranteed field'
    # Masked values are exactly zero where the mask is zero.
    assert torch.count_nonzero(out[mask_out == 0]) == 0
    print('ok: field masking keeps exactly the guaranteed field at p_keep=0')


def test_value_mask_consistency():
    torch.manual_seed(3)
    masker = DynamicMetadataMasking(p_block=0.3, p_keep_min=0.3, p_keep_max=0.7)
    meta = torch.randn(128, 13)
    out, mask_out, block_drop = masker(meta)
    # Wherever the mask is 0 the value must be 0.
    assert torch.count_nonzero(out[mask_out == 0]) == 0
    # Wherever the mask is 1 the value is unchanged.
    keep = mask_out == 1
    assert torch.allclose(out[keep], meta[keep])
    print('ok: metadata value is zeroed exactly where the mask channel is zero')


def test_model_forward_with_mask_channel():
    torch.manual_seed(4)
    model = BiDirectionalMinGRU(hidden_size=8, direction='bi', mode='parallel',
                                use_flow=False, num_meta_features=13,
                                meta_use_mask=True)
    model.eval()
    B, L = 4, 12
    x = torch.randn(B, L, 2)
    t = torch.linspace(0, 1, L).unsqueeze(0).expand(B, L).unsqueeze(-1).contiguous()
    mask = torch.ones(B, L)
    metadata = torch.randn(B, 13)
    meta_mask = torch.ones(B, 13)
    block_drop = torch.tensor([True, False, False, True])

    out = model(x, t, mask=mask, metadata=metadata, meta_mask=meta_mask,
                meta_block_drop=block_drop, return_states=True)
    assert out['h_fwd_tensor'].shape == (B, L, 8)

    # Inference path: meta_mask / meta_block_drop default to None.
    out2 = model(x, t, mask=mask, metadata=metadata, return_states=True)
    assert out2['h_fwd_tensor'].shape == (B, L, 8)
    print('ok: model.forward runs with use_mask=True (masked and inference paths)')


def test_block_drop_zeroes_metadata_contribution():
    """A block-dropped star's output must not depend on its metadata values."""
    torch.manual_seed(5)
    model = BiDirectionalMinGRU(hidden_size=8, direction='bi', mode='parallel',
                                use_flow=False, num_meta_features=13,
                                meta_use_mask=True)
    model.eval()
    B, L = 2, 10
    x = torch.randn(B, L, 2)
    t = torch.linspace(0, 1, L).unsqueeze(0).expand(B, L).unsqueeze(-1).contiguous()
    mask = torch.ones(B, L)
    meta_mask = torch.ones(B, 13)
    block_drop = torch.tensor([True, False])  # star 0 dropped, star 1 not

    meta_a = torch.randn(B, 13)
    meta_b = meta_a.clone()
    meta_b[0] = torch.randn(13)  # change ONLY the dropped star's metadata

    h_a = model(x, t, mask=mask, metadata=meta_a, meta_mask=meta_mask,
                meta_block_drop=block_drop, return_states=True)['h_fwd_tensor']
    h_b = model(x, t, mask=mask, metadata=meta_b, meta_mask=meta_mask,
                meta_block_drop=block_drop, return_states=True)['h_fwd_tensor']
    assert torch.allclose(h_a[0], h_b[0], atol=1e-6), \
        'block-dropped star output changed when its metadata changed'
    print('ok: block-dropped star ignores its metadata entirely')


if __name__ == '__main__':
    test_block_drop_all()
    test_no_masking()
    test_field_masking_keeps_at_least_one()
    test_value_mask_consistency()
    test_model_forward_with_mask_channel()
    test_block_drop_zeroes_metadata_contribution()
    print('\nAll metadata-masking tests passed.')
