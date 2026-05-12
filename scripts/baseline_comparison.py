"""Baseline comparison: RNN+flow vs. local-window predictors.

Subcommands:
    split          - deterministic gaia_id partition; writes split.json.
    train_gaussian - train MLPGaussianBaseline. One sequence per step, one
                     log-uniform k per step, all valid j positions contribute.
    eval, plot     - not implemented in v1.

Data layout assumed:
    final_pretrain/timeseries_pretrain.h5  + sidecar
    final_pretrain/timeseries_exop_hosts.h5 + sidecar

Reuses MetadataStandardizer for per-row metadata normalisation so the MLP
sees the same metadata layout as the RNN.

Run via the shell wrapper (baseline_comparison.sh) which hardcodes all
parameters; CLI args mirror those for easy debugging only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from lcgen.models.mlp_baseline import (
    MLPGaussianBaseline,
    build_context_batch,
    gaussian_nll,
    sample_k,
)
from lcgen.models.MetadataAgePredictor import DEFAULT_METADATA_FIELDS, MetadataStandardizer

METADATA_FEATURES = DEFAULT_METADATA_FIELDS


# ----------------------------- split ----------------------------------------

def _hash_int(s: str) -> int:
    """Deterministic, stable hash of a string -> non-negative int."""
    return int.from_bytes(hashlib.sha256(s.encode('utf-8')).digest()[:8], 'big')


def build_index(h5_paths):
    """Read gaia_id, length, sector, tic for every sequence across all h5 files."""
    index = []
    for p in h5_paths:
        with h5py.File(p, 'r') as f:
            lengths = f['length'][:].astype(np.int32)
            gaia = f['metadata']['GaiaDR3_ID'][:]
            tic = f['metadata']['tic'][:]
            sector = f['metadata']['sector'][:]
        for i in range(len(lengths)):
            gid = gaia[i].decode('utf-8').strip() if isinstance(gaia[i], bytes) else str(gaia[i]).strip()
            index.append({
                'file': str(p),
                'row': int(i),
                'gaia_id': gid,
                'tic_id': int(tic[i]),
                'sector': int(sector[i]),
                'length': int(lengths[i]),
            })
    return index


def make_split(h5_paths, eval_mod=10, sanity_mod=100, sanity_residue=7):
    """Partition sequences by Gaia ID.

    A star is:
        - in EVAL if hash(gid) % eval_mod == 0  (eval-10% — also stays in train, matching the RNN)
        - in SANITY-VAL if hash(gid) % sanity_mod == sanity_residue  (1%, disjoint)
        - in TRAIN if not SANITY-VAL  (=99% of stars; eval-10% included)
    """
    index = build_index(h5_paths)
    eval_ids, sanity_ids, all_ids = set(), set(), set()
    for e in index:
        g = e['gaia_id']
        all_ids.add(g)
        h = _hash_int(g)
        if h % sanity_mod == sanity_residue:
            sanity_ids.add(g)
        if h % eval_mod == 0:
            eval_ids.add(g)
    eval_ids -= sanity_ids
    return {
        'h5_paths': [str(p) for p in h5_paths],
        'eval_mod': int(eval_mod),
        'sanity_mod': int(sanity_mod),
        'sanity_residue': int(sanity_residue),
        'all_gaia_ids': sorted(all_ids),
        'eval_gaia_ids': sorted(eval_ids),
        'sanity_gaia_ids': sorted(sanity_ids),
    }


def cmd_split(args):
    split = make_split(args.h5_paths,
                       eval_mod=args.eval_mod,
                       sanity_mod=args.sanity_mod,
                       sanity_residue=args.sanity_residue)
    out = Path(args.out_dir) / 'split.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    n_all = len(split['all_gaia_ids'])
    n_eval = len(split['eval_gaia_ids'])
    n_sanity = len(split['sanity_gaia_ids'])
    n_train = n_all - n_sanity
    summary = {
        'n_all_stars': n_all,
        'n_train_stars': n_train,
        'n_eval_stars': n_eval,
        'n_sanity_stars': n_sanity,
    }
    payload = {**split, **summary}
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[split] {summary}')
    print(f'[split] wrote {out}')


# ----------------------------- data access ----------------------------------

class SequenceStore:
    """Chunk-aware H5 reader.

    The H5 files in this project are gzip-chunked at 256 rows × full-width. A
    naive per-row read forces a full chunk decompression every time (~450ms
    per row vs ~0.5ms when reading 256 rows in one slab). This store groups
    requested rows by chunk and reads each chunk once into RAM.
    """

    CHUNK_ROWS = 256  # matches the H5 chunking on disk

    def __init__(self, index, use_metadata=True, metadata_features=None, device='cpu'):
        self.index = index
        self.device = torch.device(device)
        self.use_metadata = use_metadata
        self.metadata_features = metadata_features or METADATA_FEATURES
        self.num_meta_features = len(self.metadata_features)
        self._handles = {}
        self._meta = {}
        self._max_length = {}
        files = sorted(set(e['file'] for e in index))
        std = MetadataStandardizer(fields=self.metadata_features) if use_metadata else None
        for p in files:
            with h5py.File(p, 'r') as f:
                n = int(f.attrs.get('n_samples', f['length'].shape[0]))
                self._max_length[p] = int(f['flux'].shape[1])
                if use_metadata:
                    raw = {}
                    for fld in self.metadata_features:
                        if fld in f['metadata']:
                            raw[fld] = f['metadata'][fld][:].astype(np.float32)
                        else:
                            raw[fld] = np.zeros(n, dtype=np.float32)
                    self._meta[p] = std.transform(raw)

    def _h5(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, 'r')
        return self._handles[path]

    def iter_chunks(self, indices, shuffle_chunks=False, shuffle_within=True, rng=None):
        """Yield (entry, flux, flux_err, times, metadata) tuples in chunk order.

        Reads each disk-chunk in a single slab to avoid repeated decompression.
        """
        # Bucket indices by (file, chunk_id).
        buckets = {}
        for idx in indices:
            e = self.index[idx]
            key = (e['file'], e['row'] // self.CHUNK_ROWS)
            buckets.setdefault(key, []).append(idx)
        keys = list(buckets.keys())
        if shuffle_chunks and rng is not None:
            rng.shuffle(keys)
        for key in keys:
            file = key[0]
            chunk_id = key[1]
            members = buckets[key]
            if shuffle_within and rng is not None:
                rng.shuffle(members)
            f = self._h5(file)
            row_start = chunk_id * self.CHUNK_ROWS
            row_end = row_start + self.CHUNK_ROWS
            row_end = min(row_end, f['flux'].shape[0])
            flux_chunk = f['flux'][row_start:row_end].astype(np.float32)
            ferr_chunk = f['flux_err'][row_start:row_end].astype(np.float32)
            time_chunk = f['time'][row_start:row_end].astype(np.float32)
            meta_table = self._meta.get(file) if self.use_metadata else None
            for idx in members:
                e = self.index[idx]
                local = e['row'] - row_start
                L = e['length']
                flux = torch.from_numpy(flux_chunk[local, :L].copy()).to(self.device)
                ferr = torch.from_numpy(ferr_chunk[local, :L].copy()).to(self.device)
                t = torch.from_numpy(time_chunk[local, :L].copy()).to(self.device)
                if meta_table is not None:
                    meta = torch.from_numpy(meta_table[e['row']].astype(np.float32)).to(self.device)
                else:
                    meta = None
                yield e, flux, ferr, t, meta

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()


# ----------------------------- train_gaussian -------------------------------

def _epoch_pass(model, store, indices, optimizer, K_max, C, rng,
                grad_accum=1, log_every=200, device='cpu', train=True,
                n_targets_per_seq=0):
    model.train(train)
    losses = []
    accum_loss = torch.tensor(0.0, device=device)
    accum_count = 0
    if train:
        optimizer.zero_grad()
    step = 0
    for entry, flux, flux_err, times, meta in store.iter_chunks(
            indices, shuffle_chunks=train, shuffle_within=train, rng=rng):
        L = entry['length']
        k_bound = (L - 1) // 2 - C
        if k_bound < 1:
            continue
        k = sample_k(K_max=min(K_max, k_bound), L=L, rng=rng)
        x, target_flux, _, _ = build_context_batch(
            flux, flux_err, times, meta, k=k, C=C,
            n_targets=n_targets_per_seq, rng=rng,
        )
        if x is None or x.shape[0] == 0:
            continue
        if train:
            mu, log_var = model(x)
            nll = gaussian_nll(mu, log_var, target_flux).mean()
            (nll / grad_accum).backward()
            accum_loss = accum_loss + nll.detach() * x.shape[0]
            accum_count += x.shape[0]
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                mu, log_var = model(x)
                nll = gaussian_nll(mu, log_var, target_flux).mean()
            accum_loss = accum_loss + nll.detach() * x.shape[0]
            accum_count += x.shape[0]
        losses.append(float(nll.item()))
        step += 1
        if log_every and step % log_every == 0:
            mean = float(np.mean(losses[-log_every:]))
            phase = 'train' if train else 'val'
            print(f'  [{phase}] step {step}/{len(indices)}  loss={mean:.4f}  k={k}')
    if train and (step % grad_accum) != 0:
        optimizer.step()
        optimizer.zero_grad()
    if accum_count == 0:
        return float('nan')
    return float((accum_loss / accum_count).item())


def cmd_train_gaussian(args):
    split_path = Path(args.split_path)
    if not split_path.exists():
        raise FileNotFoundError(f'split.json not found at {split_path} -- run `split` first')
    split = json.loads(split_path.read_text())

    full_index = build_index([Path(p) for p in split['h5_paths']])
    sanity_ids = set(split['sanity_gaia_ids'])

    train_idx = [i for i, e in enumerate(full_index) if e['gaia_id'] not in sanity_ids]
    sanity_idx = [i for i, e in enumerate(full_index) if e['gaia_id'] in sanity_ids]

    # Optional subsample for smoke testing.
    if args.max_train_seqs:
        train_idx = train_idx[:args.max_train_seqs]
    if args.max_sanity_seqs:
        sanity_idx = sanity_idx[:args.max_sanity_seqs]

    print(f'[train] {len(train_idx)} train sequences, {len(sanity_idx)} sanity-val sequences')

    device = torch.device(args.device)
    store = SequenceStore(full_index, use_metadata=True, device=device)

    model = MLPGaussianBaseline(
        C=args.C,
        num_meta_features=len(METADATA_FEATURES),
        hidden_dims=tuple(args.hidden_dims),
        context_dim=args.context_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train] model params: {n_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {'train_loss': [], 'sanity_loss': []}
    best_sanity = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        t0 = time.time()
        rng.shuffle(train_idx)
        train_loss = _epoch_pass(model, store, train_idx, optimizer,
                                 K_max=args.K_max, C=args.C, rng=rng,
                                 grad_accum=args.grad_accum,
                                 log_every=args.log_every, device=device,
                                 train=True,
                                 n_targets_per_seq=args.n_targets_per_seq)
        sanity_loss = _epoch_pass(model, store, sanity_idx, optimizer,
                                  K_max=args.K_max, C=args.C, rng=rng,
                                  log_every=0, device=device, train=False,
                                  n_targets_per_seq=args.n_targets_per_seq)
        dt = time.time() - t0
        history['train_loss'].append(train_loss)
        history['sanity_loss'].append(sanity_loss)
        print(f'[epoch {epoch + 1}/{args.epochs}] train_nll={train_loss:.4f}  '
              f'sanity_nll={sanity_loss:.4f}  ({dt:.1f}s)')

        ckpt = {
            'model_state': model.state_dict(),
            'config': {
                'C': args.C,
                'context_dim': args.context_dim,
                'hidden_dims': list(args.hidden_dims),
                'num_meta_features': len(METADATA_FEATURES),
                'K_max': args.K_max,
            },
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'sanity_loss': sanity_loss,
        }
        torch.save(ckpt, out_dir / 'mlp_gaussian_last.pt')
        if sanity_loss < best_sanity:
            best_sanity = sanity_loss
            best_epoch = epoch + 1
            torch.save(ckpt, out_dir / 'mlp_gaussian_best.pt')

        with open(out_dir / 'mlp_gaussian_history.json', 'w') as f:
            json.dump({
                'history': history,
                'best_epoch': best_epoch,
                'best_sanity_loss': best_sanity,
                'config': ckpt['config'],
            }, f, indent=2)

    store.close()
    print(f'[done] best epoch {best_epoch} (sanity_nll={best_sanity:.4f})')


# ----------------------------- entry ----------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('split', help='Compute deterministic gaia_id splits.')
    sp.add_argument('--h5-paths', nargs='+', required=True)
    sp.add_argument('--eval-mod', type=int, default=10)
    sp.add_argument('--sanity-mod', type=int, default=100)
    sp.add_argument('--sanity-residue', type=int, default=7)
    sp.add_argument('--out-dir', default='output/baseline_comparison')
    sp.set_defaults(func=cmd_split)

    tp = sub.add_parser('train_gaussian', help='Train Gaussian MLP baseline.')
    tp.add_argument('--split-path', default='output/baseline_comparison/split.json')
    tp.add_argument('--out-dir', default='output/baseline_comparison')
    tp.add_argument('--device', default='cpu')
    tp.add_argument('--epochs', type=int, default=5)
    tp.add_argument('--lr', type=float, default=1e-3)
    tp.add_argument('--K-max', type=int, default=720)
    tp.add_argument('--C', type=int, default=32)
    tp.add_argument('--context-dim', type=int, default=136)
    tp.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128])
    tp.add_argument('--grad-accum', type=int, default=1)
    tp.add_argument('--seed', type=int, default=0)
    tp.add_argument('--log-every', type=int, default=200)
    tp.add_argument('--max-train-seqs', type=int, default=0,
                    help='Cap training sequences for smoke tests (0 = no cap).')
    tp.add_argument('--max-sanity-seqs', type=int, default=0,
                    help='Cap sanity-val sequences for smoke tests (0 = no cap).')
    tp.add_argument('--n-targets-per-seq', type=int, default=1024,
                    help='Random subsample of valid j positions per sequence. '
                         '0 = use all valid positions (slow, full signal).')
    tp.set_defaults(func=cmd_train_gaussian)

    return p


def main():
    args = build_parser().parse_args()
    args.max_train_seqs = args.max_train_seqs if getattr(args, 'max_train_seqs', 0) > 0 else None
    args.max_sanity_seqs = args.max_sanity_seqs if getattr(args, 'max_sanity_seqs', 0) > 0 else None
    args.func(args)


if __name__ == '__main__':
    main()
