"""Unit tests for the configurable multiscale pool (glob_mode / seg_mode / diff_mode).

Covers the corrections motivated by the irregular TESS time axis:
  - diff_mode='dt' : Δt-weighted rate; mean telescopes to the net endpoint slope.
  - seg_mode='equal_time_carry' : empty time-bins filled DIRECTION-AWARE
        ([h_fwd before the gap | h_bwd after the gap]) instead of zeros.
  - seg_mode='equal_count' : equal-sample-count bins, never empty.
  - glob_mode='uniform' vs 'voronoi' differ for irregular sampling.
  - default args remain backward-compatible (voronoi / equal_time / unweighted).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_umap_latent import compute_multiscale_features  # noqa: E402

N_SEG = 4
# Block layout (each block is H wide):
# 0 glob_mean 1 glob_std 2 glob_max 3 glob_min 4-7 seg0..3 8 first 9 last
# 10 diff_mean 11 diff_std
def block(feats, idx, H):
    return feats[idx * H:(idx + 1) * H]


def test_dt_diff_mean_telescopes_to_net_slope():
    """diff_mode='dt' mean == (h_last - h_first) / (t_last - t_first), gap-robust."""
    torch.manual_seed(0)
    H = 5
    t = torch.tensor([0.0, 0.001, 0.002, 5.0, 5.001, 10.0], dtype=torch.float64)  # gaps
    h = torch.randn(len(t), H, dtype=torch.float64).cumsum(0)                      # drifting
    feats = compute_multiscale_features(h, t, n_segments=N_SEG, diff_weight_mode='dt')
    diff_mean = block(feats, 10, H)
    expected = (h[-1] - h[0]) / (t[-1] - t[0])
    assert torch.allclose(diff_mean, expected, atol=1e-9), (diff_mean, expected)


def test_equal_time_carry_is_direction_aware():
    """An empty time-bin is filled with [h_fwd[a] | h_bwd[b]] (a before, b after the gap)."""
    Hd = 3
    H = 2 * Hd  # bidirectional [fwd | bwd]
    # Cluster samples at the two ends with a big middle gap -> bins 1 and 2 empty.
    t = torch.tensor([0.0, 1.0, 2.0, 3.0, 96.0, 97.0, 98.0, 99.0, 100.0], dtype=torch.float64)
    h = torch.arange(len(t) * H, dtype=torch.float64).reshape(len(t), H)
    feats = compute_multiscale_features(
        h, t, n_segments=N_SEG, seg_mode='equal_time_carry', hidden_size=Hd,
        diff_weight_mode='dt')
    # span=100 -> edges [0,25,50,75,100]; bin0=[0,25) has t<=3 (idx0-3),
    # bins 1&2 empty, bin3=[75,100] has idx4-8. Gap is between idx 3 and 4.
    a, b = 3, 4
    expected_fill = torch.cat([h[a, :Hd], h[b, Hd:2 * Hd]], dim=0)
    seg1 = block(feats, 5, H)  # first empty bin
    seg2 = block(feats, 6, H)  # second empty bin
    assert torch.allclose(seg1, expected_fill), (seg1, expected_fill)
    assert torch.allclose(seg2, expected_fill), (seg2, expected_fill)
    # Sanity: a naive (non-direction-aware) carry would use h[a] for BOTH halves.
    naive = h[a, :]
    assert not torch.allclose(seg1, naive), "carry must differ from single-direction carry"


def test_equal_count_bins_never_empty_and_match_manual():
    """equal_count uses index quartiles; with uniform weights seg == plain mean of its rows."""
    H = 4
    t = torch.tensor([0., 1., 2., 30., 31., 32., 33., 60., 61.], dtype=torch.float64)
    h = torch.randn(len(t), H, dtype=torch.float64)
    feats = compute_multiscale_features(
        h, t, n_segments=N_SEG, seg_mode='equal_count', glob_mode='uniform',
        diff_weight_mode='dt')
    # idx_edges = round(linspace(0,9,5)) = [0,2,4,7,9] -> bins [0:2],[2:4],[4:7],[7:9]
    bins = [(0, 2), (2, 4), (4, 7), (7, 9)]
    for k, (s, e) in enumerate(bins):
        seg = block(feats, 4 + k, H)
        assert torch.allclose(seg, h[s:e].mean(0), atol=1e-9), (k, seg, h[s:e].mean(0))


def test_glob_uniform_vs_voronoi_differ_for_irregular_sampling():
    H = 3
    t = torch.tensor([0., 0.1, 0.2, 0.3, 10.0], dtype=torch.float64)  # 4 dense + 1 far
    h = torch.randn(len(t), H, dtype=torch.float64)
    f_uni = compute_multiscale_features(h, t, glob_mode='uniform', diff_weight_mode='dt')
    f_vor = compute_multiscale_features(h, t, glob_mode='voronoi', diff_weight_mode='dt')
    assert not torch.allclose(block(f_uni, 0, H), block(f_vor, 0, H)), \
        "uniform and voronoi glob_mean should differ on irregular sampling"
    # Uniform glob_mean == plain sample mean.
    assert torch.allclose(block(f_uni, 0, H), h.mean(0), atol=1e-9)


def test_voronoi_capped_limits_gap_edge_weight():
    """Capping pulls the voronoi mean toward the uniform mean (gap edges down-weighted)."""
    H = 3
    t = torch.tensor([0., 0.1, 0.2, 0.3, 50.0], dtype=torch.float64)
    h = torch.randn(len(t), H, dtype=torch.float64)
    m_uni = block(compute_multiscale_features(h, t, glob_mode='uniform', diff_weight_mode='dt'), 0, H)
    m_vor = block(compute_multiscale_features(h, t, glob_mode='voronoi', diff_weight_mode='dt'), 0, H)
    m_cap = block(compute_multiscale_features(h, t, glob_mode='voronoi_capped', diff_weight_mode='dt'), 0, H)
    d_vor = (m_vor - m_uni).abs().sum()
    d_cap = (m_cap - m_uni).abs().sum()
    assert d_cap < d_vor, (d_cap, d_vor)


def test_default_recipe_and_shape():
    """Defaults are the agreed recipe (uniform / equal_count / dt) and yield 12*H dims."""
    H = 4
    t = torch.linspace(0, 27, 50, dtype=torch.float64)
    h = torch.randn(len(t), H, dtype=torch.float64)
    f_default = compute_multiscale_features(h, t)
    f_explicit = compute_multiscale_features(
        h, t, glob_mode='uniform', seg_mode='equal_count', diff_weight_mode='dt')
    assert f_default.shape[0] == 12 * H
    assert torch.allclose(f_default, f_explicit)
    # ...and differs from the old legacy recipe.
    f_legacy = compute_multiscale_features(
        h, t, glob_mode='voronoi', seg_mode='equal_time', diff_weight_mode='unweighted')
    assert not torch.allclose(f_default, f_legacy)


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-v']))
