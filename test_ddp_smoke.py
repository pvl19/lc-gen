"""Smoke test for the DDP training path (single process, CPU).

Creates a tiny synthetic H5 file and runs 2 epochs end-to-end,
exercising: LazyH5Dataset, collate_fn, BiDirectionalMinGRU forward
(return_states=True), bounded_horizon_future_nll, checkpoint saving,
and validation.
"""
import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import h5py
import torch

# ── build a tiny synthetic H5 file ──────────────────────────────────────────
def make_h5(path, n=32, L=512):
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        f.attrs['n_samples'] = n
        f.attrs['max_length'] = L
        f.create_dataset('flux',     data=rng.normal(1.0, 0.01, (n, L)).astype(np.float32))
        f.create_dataset('flux_err', data=np.full((n, L), 0.01, dtype=np.float32))
        f.create_dataset('time',     data=np.tile(np.arange(L, dtype=np.float64), (n, 1)))
        f.create_dataset('length',   data=np.full(n, L, dtype=np.int32))
        grp = f.create_group('metadata')
        grp.create_dataset('tic',       data=np.arange(n, dtype=np.int64))
        grp.create_dataset('sector',    data=np.ones(n, dtype=np.int64))
        grp.create_dataset('exop_host', data=np.zeros(n, dtype=np.int8))
        grp.create_dataset('cadence_s', data=np.full(n, 120.0, dtype=np.float32))
        grp.create_dataset('Tmag',      data=rng.uniform(8, 14, n).astype(np.float64))
        grp.create_dataset('parallax',  data=rng.uniform(1, 10, n).astype(np.float64))
        grp.create_dataset('parallax_error', data=rng.uniform(0.01, 0.1, n).astype(np.float64))
        grp.create_dataset('G0',        data=rng.uniform(8, 14, n).astype(np.float64))
        grp.create_dataset('G0_err',    data=rng.uniform(0.001, 0.01, n).astype(np.float64))
        grp.create_dataset('BPRP0',     data=rng.uniform(0.5, 1.5, n).astype(np.float64))
        grp.create_dataset('BPRP0_err', data=rng.uniform(0.001, 0.01, n).astype(np.float64))
        grp.create_dataset('mean_flux', data=rng.uniform(1e4, 1e5, n).astype(np.float32))
        grp.create_dataset('std_flux',  data=rng.uniform(10, 1000, n).astype(np.float32))
        grp.create_dataset('camera',    data=rng.integers(1, 5, n).astype(np.float32))
        grp.create_dataset('ccd',       data=rng.integers(1, 5, n).astype(np.float32))
    print(f"  Created {path} ({n} samples, L={L})")

# ── run training ─────────────────────────────────────────────────────────────
def run():
    from lcgen.train_simple_rnn import train, parse_args
    import argparse

    with tempfile.TemporaryDirectory() as tmpdir:
        h5a = os.path.join(tmpdir, 'a.h5')
        h5b = os.path.join(tmpdir, 'b.h5')
        make_h5(h5a, n=32, L=512)
        make_h5(h5b, n=16, L=256)

        # Fake args matching the SLURM config (small scale)
        args = argparse.Namespace(
            input=[h5a, h5b],
            direction='bi',
            random_seed=42,
            num_samples=None,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            hidden_size=32,
            output_dir=os.path.join(tmpdir, 'output'),
            output_name='model.pt',
            device='auto',
            use_flow=True,
            min_size=5,
            max_size=64,
            mask_portion=0.4,
            mock_sinusoid=False,
            mock_noise=0.1,
            mode='parallel',
            K=32,
            k_spacing='log',
            use_metadata=True,
            use_conv_channels=False,
            conv_encoder_type='unet',
            val_split=0.1,
            val_k_values=[1, 2, 4, 8],
            patience=10,
            min_delta=0.0,
            resume_from=None,
            save_every=1,
            checkpoint_copy_dir=None,
        )

        print("\nRunning train()...")
        train(args)
        print("\nSmoke test PASSED")

if __name__ == '__main__':
    run()
