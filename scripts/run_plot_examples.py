#!/usr/bin/env python3
"""
Run plot_reconstruction_examples for a small set and save outputs to /tmp.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_rnn_autoencoder import load_lightcurve_data, LightCurveDataset, collate_with_masking, plot_reconstruction_examples
from scripts.demo_sample_flow import load_model

MODEL_PATH = Path('tmp_fulltrain_trunc2048_logprob/rnn_final.pt')
H5_PATH = Path('data/real_lightcurves/timeseries.h5')
# Save plots and npz inside the project tmp/ directory (not root /tmp)
OUT_PNG = Path('tmp/reconstruction_examples.png')

N_EXAMPLES = 4
NUM_SAMPLES = 4


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if not MODEL_PATH.exists():
        print('Model file not found:', MODEL_PATH)
        return
    if not H5_PATH.exists():
        print('HDF5 data not found:', H5_PATH)
        return

    model = load_model(device)

    flux, timestamps, flux_err = load_lightcurve_data(str(H5_PATH))
    # create a small dataset with first N_EXAMPLES
    flux_small = flux[:N_EXAMPLES]
    time_small = timestamps[:N_EXAMPLES]

    dataset = LightCurveDataset(flux_small, time_small)
    dataset.flux_err = torch.from_numpy(flux_err[:N_EXAMPLES]).float()

    # collate function with reasonable masking params
    collate_fn = partial(collate_with_masking, min_block_size=1, max_block_size=32, min_mask_ratio=0.1, max_mask_ratio=0.5)

    loader = DataLoader(dataset, batch_size=N_EXAMPLES, shuffle=False, collate_fn=collate_fn)

    # Call plotting helper (this will save images and optionally save example .npz)
    print('Calling plot_reconstruction_examples...')
    plot_reconstruction_examples(model, loader, device, str(OUT_PNG), n_examples=N_EXAMPLES, sample_eval=True, num_samples=NUM_SAMPLES, base_seed=0, eval_teacher_forcing=True, save_examples_data=True)

if __name__ == '__main__':
    main()
