"""Re-extract latents for sectors 97 and 98 with each ~60-day light curve split
in half.

Sectors 97 and 98 are ~60-day observations (~2x the typical 27-day TESS sector
length), which makes the multiscale pool sit on a different sequence-length
regime than the rest of the dataset. This script:

  1. Loads the trained model.
  2. For each H5 file, finds rows in the target sectors (default 97, 98).
  3. Splits each row's *post-trim* valid range in half at its midpoint.
  4. Runs the encoder on each half independently (with the same metadata).
  5. Computes the same multiscale features as plot_umap_latent.py.
  6. Saves a sidecar npz alongside the existing latents (does NOT overwrite).

The output npz is in the canonical kfold/UMAP format
(latent_vectors, ages, bprp0, gaia_ids, tic_ids, sectors + extras) with one
extra ``subsector`` field (0 = first half, 1 = second half). Each original row
becomes two consecutive output rows with the same gaia_id / tic_id / sector and
the same age/bprp0/etc. metadata — only the latent and subsector differ.

To later merge: drop sector 97/98 rows from the main npz, then concatenate
these split rows.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcgen.models.TimeSeriesDataset import METADATA_FEATURES
from lcgen.models.MetadataAgePredictor import MetadataStandardizer

from plot_umap_latent import (
    load_model,
    compute_multiscale_features,
    load_ages,
)


def build_split_records(lengths, target_idx, trim_edges, min_post):
    """For each target row, build (orig_h5_idx, half_idx, src_start, src_stop)
    records — two per original row. Rows with post-trim valid length below
    2 * min_post (so each half has at least min_post samples) are dropped.
    """
    records = []
    n_dropped = 0
    for h5_i in target_idx:
        L = int(lengths[h5_i])
        valid = L - 2 * trim_edges
        if valid < 2 * min_post:
            n_dropped += 1
            continue
        mid = valid // 2
        a_start = trim_edges
        a_stop = trim_edges + mid
        b_start = trim_edges + mid
        b_stop = trim_edges + valid
        records.append((int(h5_i), 0, a_start, a_stop))
        records.append((int(h5_i), 1, b_start, b_stop))
    return records, n_dropped


def load_metadata_for_indices(h5_path, h5_indices, zero_metadata=False,
                              zero_fields=None):
    """Return (metadata_all, meta_mask_all) for the given h5 row indices in the
    order requested, or (None, None) if the H5 has no metadata group. Standard
    DOROTHY-style mask channel: ones by default, zeroed columns where the field
    is withheld.
    """
    with h5py.File(h5_path, 'r') as f:
        if 'metadata' not in f:
            return None, None
        meta_grp = f['metadata']
        sorted_idx = np.sort(h5_indices)
        order = np.argsort(np.argsort(h5_indices))  # inverse permutation
        n = len(sorted_idx)
        raw = {}
        for feat in METADATA_FEATURES:
            if feat in meta_grp:
                vals = meta_grp[feat][sorted_idx]
                raw[feat] = vals.astype(np.float32)
            else:
                raw[feat] = np.zeros(n, dtype=np.float32)
    standardizer = MetadataStandardizer(fields=METADATA_FEATURES)
    metadata_sorted = standardizer.transform(raw)
    metadata_all = metadata_sorted[order]
    meta_mask_all = np.ones_like(metadata_all)
    zero_cols = []
    if zero_metadata:
        zero_cols = list(range(len(METADATA_FEATURES)))
    elif zero_fields:
        zero_cols = [METADATA_FEATURES.index(f) for f in zero_fields
                     if f in METADATA_FEATURES]
    if zero_cols:
        metadata_all[:, zero_cols] = 0.0
        meta_mask_all[:, zero_cols] = 0.0
    return metadata_all, meta_mask_all


def apply_head_norm_to_states(model, h_fwd, h_bwd, t_enc):
    """Mirror plot_umap_latent.extract_latent_vectors' apply_head_norm branch:
    reconstruct the tensor [h_fwd ‖ h_bwd ‖ t_enc] the head_norm LayerNorm was
    trained on, normalize, then slice the hidden parts back out.
    """
    if not hasattr(model, 'head_norm') or model.head_norm is None:
        raise ValueError("apply_head_norm=True but model has no head_norm module")
    if t_enc is None:
        raise ValueError("apply_head_norm requires t_enc in model output")
    H = model.hidden_size
    if h_fwd is not None and h_bwd is not None:
        h_bi = torch.cat([h_fwd, h_bwd, t_enc], dim=-1)
    elif h_fwd is not None:
        h_bi = torch.cat([h_fwd, t_enc], dim=-1)
    elif h_bwd is not None:
        h_bi = torch.cat([h_bwd, t_enc], dim=-1)
    else:
        raise ValueError("No hidden states returned by model")
    h_bi = model.head_norm(h_bi)
    if h_fwd is not None and h_bwd is not None:
        return h_bi[..., :H], h_bi[..., H:2 * H]
    if h_fwd is not None:
        return h_bi[..., :H], None
    return None, h_bi[..., :H]


def extract_split_latents(model, h5_path, target_sectors, device, *,
                          trim_edges, batch_size, use_metadata, zero_metadata,
                          zero_fields, apply_head_norm, minmax_edge_skip,
                          diff_weight_mode, minmax_quantile,
                          subtract_temporal_mean, glob_mode, seg_mode,
                          min_post=32):
    """Returns:
        latents: (2N_kept, 12*2*H) array
        record_h5_idx: (2N_kept,) int64 — original H5 row index per output row
        record_half:   (2N_kept,) int8  — 0 or 1
        kept_orig_idx: (N_kept,)  int64 — unique original H5 indices kept
    """
    with h5py.File(h5_path, 'r') as f:
        sectors_all = f['metadata']['sector'][:]
        lengths_all = f['length'][:]
    target_mask = np.isin(sectors_all, np.asarray(target_sectors, dtype=sectors_all.dtype))
    target_idx = np.where(target_mask)[0]
    print(f'  Found {len(target_idx)} rows in sectors {list(target_sectors)} '
          f'(out of {len(sectors_all)} total)')

    records, n_dropped = build_split_records(lengths_all, target_idx,
                                             trim_edges, min_post)
    if n_dropped:
        print(f'  Dropped {n_dropped} rows with post-trim valid length < {2 * min_post} '
              f'(each half would have <{min_post} samples)')
    if not records:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.int64))

    # Load metadata for the unique original rows once, then look up per item.
    kept_orig = sorted({rec[0] for rec in records})
    kept_orig_arr = np.asarray(kept_orig, dtype=np.int64)
    orig_to_meta_row = {h5_i: j for j, h5_i in enumerate(kept_orig_arr)}

    load_meta = use_metadata and (model.meta_encoder is not None)
    if load_meta:
        metadata_all, meta_mask_all = load_metadata_for_indices(
            h5_path, kept_orig_arr, zero_metadata=zero_metadata,
            zero_fields=zero_fields)
        if metadata_all is None:
            load_meta = False

    n_recs = len(records)
    latents = []
    rec_h5_idx = np.array([r[0] for r in records], dtype=np.int64)
    rec_half = np.array([r[1] for r in records], dtype=np.int8)

    print(f'  Extracting {n_recs} half-latents in batches of {batch_size}...')

    with h5py.File(h5_path, 'r') as f, torch.no_grad():
        for start in range(0, n_recs, batch_size):
            chunk = records[start:start + batch_size]
            seg_lengths = np.array([stop - st for (_, _, st, stop) in chunk],
                                   dtype=np.int64)
            B = len(chunk)
            L = int(seg_lengths.max())

            flux = torch.zeros(B, L, dtype=torch.float32, device=device)
            ferr = torch.zeros(B, L, dtype=torch.float32, device=device)
            tm   = torch.zeros(B, L, dtype=torch.float32, device=device)
            mask = torch.zeros(B, L, device=device)

            for j, (h5_i, _hi, st, stop) in enumerate(chunk):
                ll = stop - st
                flux[j, :ll] = torch.tensor(f['flux'][h5_i, st:stop],
                                            dtype=torch.float32, device=device)
                ferr[j, :ll] = torch.tensor(f['flux_err'][h5_i, st:stop],
                                            dtype=torch.float32, device=device)
                tm[j, :ll]   = torch.tensor(f['time'][h5_i, st:stop],
                                            dtype=torch.float32, device=device)
                mask[j, :ll] = 1.0

            x_in = torch.stack([flux, ferr], dim=-1)
            t_in = tm.unsqueeze(-1)

            if load_meta:
                meta_rows = np.array(
                    [orig_to_meta_row[r[0]] for r in chunk], dtype=np.int64)
                meta_batch = torch.tensor(metadata_all[meta_rows],
                                          dtype=torch.float32, device=device)
                meta_mask_batch = torch.tensor(meta_mask_all[meta_rows],
                                               dtype=torch.float32, device=device)
            else:
                meta_batch = None
                meta_mask_batch = None

            out = model(x_in, t_in, mask=mask,
                        metadata=meta_batch, meta_mask=meta_mask_batch,
                        conv_data=None, return_states=True)
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')
            t_enc = out.get('t_enc')

            if apply_head_norm:
                h_fwd, h_bwd = apply_head_norm_to_states(model, h_fwd, h_bwd, t_enc)

            for i in range(B):
                vl = int(seg_lengths[i])
                if h_fwd is not None and h_bwd is not None:
                    h_combined = torch.cat([h_fwd[i, :vl, :], h_bwd[i, :vl, :]], dim=-1)
                elif h_fwd is not None:
                    h_combined = h_fwd[i, :vl, :]
                elif h_bwd is not None:
                    h_combined = h_bwd[i, :vl, :]
                else:
                    raise ValueError("No hidden states returned by model")
                t_combined = tm[i, :vl]
                latent = compute_multiscale_features(
                    h_combined, t_combined, n_segments=4,
                    minmax_edge_skip=minmax_edge_skip,
                    diff_weight_mode=diff_weight_mode,
                    minmax_quantile=minmax_quantile,
                    subtract_temporal_mean=subtract_temporal_mean,
                    hidden_size=model.hidden_size,
                    glob_mode=glob_mode, seg_mode=seg_mode,
                )
                latents.append(latent.cpu().numpy())

            if (start // batch_size) % 10 == 0:
                print(f'    {min(start + batch_size, n_recs)}/{n_recs}')

    latents = np.stack(latents, axis=0)
    print(f'  Extracted half-latents with shape {latents.shape}')
    return latents, rec_h5_idx, rec_half, kept_orig_arr


def save_split_npz(out_path, latents, rec_h5_idx, rec_half,
                   ages, bprp0, bprp0_err, mg, mg_err, mem_prob,
                   gaia_ids, tic_ids, sectors):
    """Save in canonical kfold/UMAP format + extra `subsector` field. Each
    record's ID/metadata fields are looked up via its original H5 row index.
    """
    n = len(rec_h5_idx)
    # Note: load_ages returns arrays in H5 row order (sorted indices when given),
    # but here we passed sample_indices=None so they map 1:1 to h5 row index.
    out = dict(
        latent_vectors=np.asarray(latents, dtype=np.float32),
        ages=ages[rec_h5_idx],
        bprp0=bprp0[rec_h5_idx],
        bprp0_err=bprp0_err[rec_h5_idx],
        mg=mg[rec_h5_idx],
        mg_err=mg_err[rec_h5_idx],
        mem_prob=mem_prob[rec_h5_idx],
        gaia_ids=gaia_ids[rec_h5_idx].astype(str),
        tic_ids=tic_ids[rec_h5_idx].astype(np.int64),
        sectors=sectors[rec_h5_idx].astype(np.int64),
        subsector=np.asarray(rec_half, dtype=np.int8),
        orig_h5_idx=np.asarray(rec_h5_idx, dtype=np.int64),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f'  Saved {n} split-half latents -> {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', required=True)
    p.add_argument('--h5_path', nargs='+', required=True)
    p.add_argument('--age_csv_path', nargs='+', required=True,
                   help='Same CSVs as plot_umap_latent.py (e.g. metadata.csv host_all_metadata.csv).')
    p.add_argument('--save_split_latents', nargs='+', required=True,
                   help='Output npz path per H5 (same order as --h5_path).')
    p.add_argument('--target_sectors', type=int, nargs='+', default=[97, 98])

    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--direction', default='bi', choices=['forward', 'backward', 'bi'])
    p.add_argument('--mode', default='parallel', choices=['sequential', 'parallel'])
    p.add_argument('--use_flow', action='store_true')

    p.add_argument('--use_metadata', action='store_true')
    p.add_argument('--zero_metadata', action='store_true')
    p.add_argument('--zero_fields', default=None,
                   help='Comma-separated metadata field names to withhold.')

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--trim_edges', type=int, default=10)
    p.add_argument('--minmax_edge_skip', type=int, default=0)
    p.add_argument('--diff_weight_mode', default='dt',
                   choices=['dt', 'unweighted', 'dt_inverse'])
    p.add_argument('--glob_mode', default='uniform',
                   choices=['uniform', 'voronoi', 'voronoi_capped'])
    p.add_argument('--seg_mode', default='equal_count',
                   choices=['equal_count', 'equal_time', 'equal_time_carry'])
    p.add_argument('--minmax_quantile', type=float, default=0.0)
    p.add_argument('--subtract_temporal_mean', action='store_true')
    p.add_argument('--apply_head_norm', action=argparse.BooleanOptionalAction,
                   default=True)
    args = p.parse_args()

    if len(args.save_split_latents) != len(args.h5_path):
        p.error('--save_split_latents must have one path per --h5_path')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_meta_features = len(METADATA_FEATURES) if args.use_metadata else 0
    model = load_model(
        args.model_path, device,
        hidden_size=args.hidden_size,
        direction=args.direction,
        mode=args.mode,
        use_flow=args.use_flow,
        num_meta_features=num_meta_features,
        use_conv_channels=False,
        conv_config=None,
    )

    zero_fields = args.zero_fields.split(',') if args.zero_fields else None

    for h5_path, out_path in zip(args.h5_path, args.save_split_latents):
        print(f'\n--- {h5_path} -> {out_path} ---')
        latents, rec_h5_idx, rec_half, kept = extract_split_latents(
            model, h5_path, args.target_sectors, device,
            trim_edges=args.trim_edges,
            batch_size=args.batch_size,
            use_metadata=args.use_metadata,
            zero_metadata=args.zero_metadata,
            zero_fields=zero_fields,
            apply_head_norm=args.apply_head_norm,
            minmax_edge_skip=args.minmax_edge_skip,
            diff_weight_mode=args.diff_weight_mode,
            minmax_quantile=args.minmax_quantile,
            subtract_temporal_mean=args.subtract_temporal_mean,
            glob_mode=args.glob_mode,
            seg_mode=args.seg_mode,
        )
        if latents.size == 0:
            print('  No rows in target sectors; nothing to save.')
            continue

        ages, bprp0, bprp0_err, mg, mg_err, mem_prob, gaia_ids, tic_ids, sectors = \
            load_ages(h5_path, args.age_csv_path, sample_indices=None)

        save_split_npz(out_path, latents, rec_h5_idx, rec_half,
                       ages, bprp0, bprp0_err, mg, mg_err, mem_prob,
                       gaia_ids, tic_ids, sectors)

    print('\nDone.')


if __name__ == '__main__':
    main()
