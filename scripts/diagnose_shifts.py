import numpy as np
import json
from pathlib import Path

def best_shift_pair(a, b, max_shift=64):
    # a, b: 1D arrays same length L
    L = a.shape[0]
    best_s = None
    best_mse = float('inf')
    best_corr = 0.0
    for s in range(-max_shift, max_shift+1):
        # align overlapping region
        if s >= 0:
            a_seg = a[:L-s]
            b_seg = b[s:]
        else:
            a_seg = a[-s:]
            b_seg = b[:L+s]
        if a_seg.size == 0:
            continue
        mse = float(np.nanmean((a_seg - b_seg)**2))
        if mse < best_mse:
            best_mse = mse
            best_s = s
            # compute corr on overlapping, mean-subtracted
            aa = a_seg - np.nanmean(a_seg)
            bb = b_seg - np.nanmean(b_seg)
            denom = (np.nanstd(aa) * np.nanstd(bb))
            if denom > 0:
                best_corr = float(np.nansum(aa*bb) / (aa.size * np.nanstd(aa) * np.nanstd(bb)))
            else:
                best_corr = 0.0
    return best_s, best_mse, best_corr

def analyze(npz_path, ks=(1,8,32,128), max_shift=64):
    data = np.load(npz_path)
    ks_in = list(data['ks'])
    base_k = ks[0]
    out = {}
    for k in ks[1:]:
        key_a = f'pred_k_{base_k}'
        key_b = f'pred_k_{k}'
        if key_a not in data or key_b not in data:
            continue
        A = data[key_a]  # shape (B, L)
        B = data[key_b]
        Bn = A.shape[0]
        pair_res = []
        for i in range(Bn):
            a = A[i]
            b = B[i]
            s, mse, corr = best_shift_pair(a, b, max_shift=max_shift)
            # compute residual std after shifting
            L = a.shape[0]
            if s >= 0:
                a_seg = a[:L-s]
                b_seg = b[s:]
            else:
                a_seg = a[-s:]
                b_seg = b[:L+s]
            res = a_seg - b_seg
            res_std = float(np.nanstd(res))
            orig_std = float(np.nanstd(a_seg))
            frac_var_expl = 1.0 - (res_std**2) / (orig_std**2 + 1e-12)
            pair_res.append({'shift': int(s), 'mse': float(mse), 'corr': float(corr), 'res_std': res_std, 'orig_std': orig_std, 'frac_var_explained': float(frac_var_expl)})
        # aggregate
        shifts = [p['shift'] for p in pair_res]
        res_std_mean = float(np.mean([p['res_std'] for p in pair_res]))
        orig_std_mean = float(np.mean([p['orig_std'] for p in pair_res]))
        frac_mean = float(np.mean([p['frac_var_explained'] for p in pair_res]))
        out[f'{base_k}_vs_{k}'] = {'per_example': pair_res, 'median_shift': int(np.median(shifts)), 'mean_residual_std': res_std_mean, 'mean_orig_std': orig_std_mean, 'mean_frac_var_explained': frac_mean}
    return out

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--npz', type=str, default='output/simple_rnn/recon_data/reconstructions_v11_multipred_fwd_data.npz')
    p.add_argument('--out_json', type=str, default='output/simple_rnn/recon_data/recon_k_shift_diag.json')
    p.add_argument('--max_shift', type=int, default=64)
    args = p.parse_args()
    res = analyze(args.npz, ks=(1,8,32,128), max_shift=args.max_shift)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(res, fh, indent=2)
    print('Wrote diagnostics to', args.out_json)
