import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_rel(npz_path, out_png, num_examples=3):
    data = np.load(npz_path)
    ks = list(data['ks'])
    flux = data['flux']
    times = data['times']
    flux_err = data['flux_err']
    B, L = flux.shape

    # collect relative preds keys
    preds_rel = {}
    for k in ks:
        key = f'pred_k_rel_{k}'
        if key in data:
            preds_rel[k] = data[key]

    nplot = min(num_examples, B)
    fig, axes = plt.subplots(nplot, 1, figsize=(14, nplot * 3))
    if nplot == 1:
        axes = [axes]

    cmap = plt.get_cmap('tab10')
    for i in range(nplot):
        ax = axes[i]
        ax.errorbar(times[i], flux[i], yerr=flux_err[i], fmt='.', color='k', alpha=0.6, label='Original')
        for idx, (k, arr) in enumerate(sorted(preds_rel.items())):
            pred_i = arr[i]
            color = cmap(idx % 10)
            ax.plot(times[i], pred_i, label=f'Pred (rel) from t-{k}', color=color, linestyle='--')
        ax.set_title(f'Example {i+1} - Relative-time predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.legend()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    print('Saved relative-only plot to', out_png)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--npz', type=str, default='output/simple_rnn/recon_data/reconstructions_v11_multipred_fwd_rel_data.npz')
    p.add_argument('--out', type=str, default='output/simple_rnn/recon_plots/reconstructions_v11_multipred_fwd_rel.png')
    p.add_argument('--num_examples', type=int, default=3)
    args = p.parse_args()
    plot_rel(args.npz, args.out, num_examples=args.num_examples)
