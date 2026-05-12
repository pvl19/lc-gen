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


# ----------------------------- eval -----------------------------------------

EVAL_K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _load_rnn_model(model_path: str, device, num_meta_features: int):
    """Load the trained BiDirectionalMinGRU + flow head."""
    from lcgen.models.simple_min_gru import BiDirectionalMinGRU
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    has_meta = any(k.startswith('meta_encoder.') for k in sd)
    has_flow = any(k.startswith('flow.') for k in sd)
    has_conv = any(k.startswith('ps_encoder.') for k in sd)
    if not has_meta:
        num_meta_features = 0
    if isinstance(ckpt, dict) and 'num_meta_features' in ckpt:
        num_meta_features = ckpt['num_meta_features']
    model = BiDirectionalMinGRU(
        hidden_size=64, direction='bi', mode='parallel',
        use_flow=has_flow, num_meta_features=num_meta_features,
        use_conv_channels=has_conv,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f'[rnn] loaded {model_path} (meta={num_meta_features}, conv={has_conv}, flow={has_flow})')
    return model


def _rnn_forward_cache(rnn, flux, ferr, times, meta):
    """Single forward over a sequence; returns h_fwd, h_bwd, t_enc."""
    flux_b = flux.unsqueeze(0)
    ferr_b = ferr.unsqueeze(0)
    times_b = times.unsqueeze(0)
    L = flux_b.size(1)
    mask = torch.ones(1, L, device=flux.device)
    x_in = torch.stack([flux_b, ferr_b], dim=-1)
    t_in = times_b.unsqueeze(-1)
    meta_b = meta.unsqueeze(0) if meta is not None else None
    out = rnn(x_in, t_in, mask=mask, metadata=meta_b, conv_data=None, return_states=True)
    return {
        'h_fwd': out.get('h_fwd_tensor'),
        'h_bwd': out.get('h_bwd_tensor'),
        't_enc': out['t_enc'],
    }


def _rnn_flow_predict(rnn, cache, js, k, ferr, target, n_samples: int):
    """Evaluate flow head at (js, k). Returns (log_prob, samples or None)."""
    h_fwd = cache['h_fwd']
    h_bwd = cache['h_bwd']
    t_enc = cache['t_enc']
    src_f = h_fwd[0, js - k]
    src_b = h_bwd[0, js + k]
    t_tgt = t_enc[0, js]
    Te = t_tgt.size(-1)
    flat_in = torch.cat([src_f, src_b, t_tgt], dim=-1)
    normed = rnn.head_norm(flat_in)
    if Te > 0:
        normed = torch.cat([normed[:, :-Te], normed[:, -Te:] * rnn.time_scale], dim=1)
    ferr_j = ferr[js].unsqueeze(-1)
    ctx = torch.cat([normed, ferr_j], dim=-1)
    dist = rnn.flow(ctx)
    log_prob = dist.log_prob(target.unsqueeze(-1)).view(-1)
    samples = None
    if n_samples > 0:
        s = torch.stack([dist.sample() for _ in range(n_samples)], dim=0).squeeze(-1)
        samples = s  # (S, N)
    return log_prob, samples


class _StatAcc:
    """Running sums for MAE / RMSE / NLL / coverage per (method, k)."""

    def __init__(self):
        self.bins = {}

    def _bin(self, key):
        s = self.bins.get(key)
        if s is None:
            s = {'n': 0, 'sum_ae': 0.0, 'sum_se': 0.0,
                 'sum_nll': 0.0, 'n_nll': 0,
                 'n_c68': 0, 'n_c95': 0, 'n_cov': 0, 'seqs': set()}
            self.bins[key] = s
        return s

    def add_point(self, method, k, target, pred, seq_id=None):
        s = self._bin((method, k))
        err = (target - pred)
        s['n'] += int(target.numel())
        s['sum_ae'] += float(err.abs().sum().item())
        s['sum_se'] += float((err ** 2).sum().item())
        if seq_id is not None:
            s['seqs'].add(seq_id)

    def add_gaussian(self, method, k, target, mu, sigma, seq_id=None):
        s = self._bin((method, k))
        err = (target - mu)
        s['n'] += int(target.numel())
        s['sum_ae'] += float(err.abs().sum().item())
        s['sum_se'] += float((err ** 2).sum().item())
        nll = 0.5 * (torch.log(2 * math.pi * sigma ** 2) + (err / sigma) ** 2)
        s['sum_nll'] += float(nll.sum().item())
        s['n_nll'] += int(target.numel())
        z = err / sigma
        s['n_c68'] += int((z.abs() <= 1.0).sum().item())
        s['n_c95'] += int((z.abs() <= 1.96).sum().item())
        s['n_cov'] += int(target.numel())
        if seq_id is not None:
            s['seqs'].add(seq_id)

    def add_flow(self, method, k, target, log_prob, samples, seq_id=None):
        s = self._bin((method, k))
        s['sum_nll'] += float((-log_prob).sum().item())
        s['n_nll'] += int(target.numel())
        if samples is not None:
            median = samples.quantile(0.5, dim=0)
            p16 = samples.quantile(0.16, dim=0)
            p84 = samples.quantile(0.84, dim=0)
            p025 = samples.quantile(0.025, dim=0)
            p975 = samples.quantile(0.975, dim=0)
            err = (target - median)
            s['n'] += int(target.numel())
            s['sum_ae'] += float(err.abs().sum().item())
            s['sum_se'] += float((err ** 2).sum().item())
            in68 = ((target >= p16) & (target <= p84))
            in95 = ((target >= p025) & (target <= p975))
            s['n_c68'] += int(in68.sum().item())
            s['n_c95'] += int(in95.sum().item())
            s['n_cov'] += int(target.numel())
        if seq_id is not None:
            s['seqs'].add(seq_id)

    def rows(self):
        out = []
        for (method, k), s in sorted(self.bins.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            n = max(s['n'], 1)
            n_nll = s['n_nll']
            n_cov = s['n_cov']
            row = {
                'method': method,
                'k': k,
                'n_points': s['n'],
                'n_seqs': len(s['seqs']),
                'mae': s['sum_ae'] / n if s['n'] else float('nan'),
                'rmse': math.sqrt(s['sum_se'] / n) if s['n'] else float('nan'),
                'nll': (s['sum_nll'] / n_nll) if n_nll else float('nan'),
                'coverage68': (s['n_c68'] / n_cov) if n_cov else float('nan'),
                'coverage95': (s['n_c95'] / n_cov) if n_cov else float('nan'),
            }
            out.append(row)
        return out


def cmd_eval(args):
    split = json.loads(Path(args.split_path).read_text())
    full_index = build_index([Path(p) for p in split['h5_paths']])
    eval_ids = set(split['eval_gaia_ids'])
    eval_idx = [i for i, e in enumerate(full_index) if e['gaia_id'] in eval_ids]
    if args.max_eval_seqs:
        eval_idx = eval_idx[:args.max_eval_seqs]
    print(f'[eval] {len(eval_idx)} eval sequences')

    device = torch.device(args.device)
    store = SequenceStore(full_index, use_metadata=True, device=device)

    # Load MLP Gaussian.
    mlp_ckpt = torch.load(args.mlp_path, map_location=device, weights_only=False)
    cfg = mlp_ckpt['config']
    C = int(cfg['C'])
    mlp = MLPGaussianBaseline(
        C=C,
        num_meta_features=int(cfg['num_meta_features']),
        hidden_dims=tuple(cfg['hidden_dims']),
        context_dim=int(cfg['context_dim']),
    ).to(device)
    mlp.load_state_dict(mlp_ckpt['model_state'])
    mlp.eval()
    print(f'[mlp] loaded {args.mlp_path} (C={C}, context_dim={cfg["context_dim"]})')

    # Load RNN (optional).
    rnn = None
    if args.rnn_path:
        rnn = _load_rnn_model(args.rnn_path, device,
                              num_meta_features=len(METADATA_FEATURES))

    rng = np.random.default_rng(args.seed)
    acc = _StatAcc()
    arange_C = torch.arange(C, device=device, dtype=torch.long)

    t0 = time.time()
    seq_count = 0
    with torch.no_grad():
        for entry, flux, ferr, times, meta in store.iter_chunks(
                eval_idx, shuffle_chunks=False, shuffle_within=False):
            L = entry['length']
            seq_id = (entry['gaia_id'], entry['sector'])

            rnn_cache = None
            if rnn is not None:
                rnn_cache = _rnn_forward_cache(rnn, flux, ferr, times, meta)

            for k in EVAL_K_GRID:
                j_min = k + C
                j_max = L - k - C
                if j_max <= j_min:
                    continue
                if k > args.K_max:
                    continue

                n_full = j_max - j_min
                if args.n_eval_targets_per_seq and args.n_eval_targets_per_seq < n_full:
                    picks = rng.choice(n_full, size=args.n_eval_targets_per_seq, replace=False)
                    picks.sort()
                    js = torch.from_numpy(picks.astype(np.int64) + j_min).to(device)
                else:
                    js = torch.arange(j_min, j_max, device=device, dtype=torch.long)

                target = flux[js]

                # MLP Gaussian.
                x, _, _, _ = build_context_batch(
                    flux, ferr, times, meta, k=k, C=C, js=js,
                )
                mu, log_var = mlp(x)
                log_var = log_var.clamp(-10.0, 6.0)
                sigma = torch.exp(0.5 * log_var)
                acc.add_gaussian('mlp_gaussian', k, target, mu, sigma, seq_id=seq_id)

                # NN-mean (== linear interp midpoint).
                pred_nn = 0.5 * (flux[js - k] + flux[js + k])
                acc.add_point('nn_mean', k, target, pred_nn, seq_id=seq_id)

                # Window-mean.
                fwd_idx = js.unsqueeze(1) - k - C + arange_C.unsqueeze(0)
                bwd_idx = js.unsqueeze(1) + k + arange_C.unsqueeze(0)
                fwd = flux[fwd_idx].mean(dim=1)
                bwd = flux[bwd_idx].mean(dim=1)
                pred_wm = 0.5 * (fwd + bwd)
                acc.add_point('window_mean', k, target, pred_wm, seq_id=seq_id)

                # RNN flow.
                if rnn_cache is not None:
                    log_prob, samples = _rnn_flow_predict(
                        rnn, rnn_cache, js, k, ferr, target,
                        n_samples=args.n_flow_samples,
                    )
                    acc.add_flow('rnn_flow', k, target, log_prob, samples,
                                 seq_id=seq_id)

            seq_count += 1
            if args.log_every and seq_count % args.log_every == 0:
                dt = time.time() - t0
                rate = seq_count / dt if dt > 0 else 0.0
                print(f'  [eval] {seq_count}/{len(eval_idx)} seqs  ({rate:.2f} seq/s)')

    store.close()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = acc.rows()
    csv_path = out_dir / 'summary.csv'
    cols = ['method', 'k', 'n_points', 'n_seqs', 'mae', 'rmse', 'nll',
            'coverage68', 'coverage95']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'[eval] wrote {csv_path} ({len(rows)} rows)')
    # Pretty-print so the user can read the result immediately.
    by_method = {}
    for r in rows:
        by_method.setdefault(r['method'], []).append(r)
    for method, rs in sorted(by_method.items()):
        rs.sort(key=lambda x: x['k'])
        print(f'  {method}')
        print(f'    k     MAE      RMSE     NLL       cov68   cov95')
        for r in rs:
            nll = '   nan ' if math.isnan(r['nll']) else f'{r["nll"]:7.4f}'
            c68 = '  nan' if math.isnan(r['coverage68']) else f'{r["coverage68"]:5.3f}'
            c95 = '  nan' if math.isnan(r['coverage95']) else f'{r["coverage95"]:5.3f}'
            print(f'    {r["k"]:<4d}  {r["mae"]:7.4f}  {r["rmse"]:7.4f}  {nll}   {c68}   {c95}')


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

    ep = sub.add_parser('eval', help='Evaluate baselines on the eval-10% split.')
    ep.add_argument('--split-path', default='output/baseline_comparison/split.json')
    ep.add_argument('--out-dir', default='output/baseline_comparison')
    ep.add_argument('--mlp-path', required=True,
                    help='Path to mlp_gaussian_best.pt')
    ep.add_argument('--rnn-path', default=None,
                    help='Path to the RNN+flow checkpoint (e.g. final_model/parallel_fixed/e110/best_model.pt). '
                         'If omitted, only MLP + non-learned baselines are evaluated.')
    ep.add_argument('--device', default='cpu')
    ep.add_argument('--seed', type=int, default=0)
    ep.add_argument('--K-max', type=int, default=720)
    ep.add_argument('--n-eval-targets-per-seq', type=int, default=256,
                    help='Subsample of valid j per (sequence, k). 0 = all valid positions.')
    ep.add_argument('--n-flow-samples', type=int, default=0,
                    help='Flow samples per point for MAE/coverage. 0 = NLL only (cheap).')
    ep.add_argument('--max-eval-seqs', type=int, default=0)
    ep.add_argument('--log-every', type=int, default=100)
    ep.set_defaults(func=cmd_eval)

    return p


def main():
    args = build_parser().parse_args()
    args.max_train_seqs = args.max_train_seqs if getattr(args, 'max_train_seqs', 0) > 0 else None
    args.max_sanity_seqs = args.max_sanity_seqs if getattr(args, 'max_sanity_seqs', 0) > 0 else None
    if hasattr(args, 'max_eval_seqs'):
        args.max_eval_seqs = args.max_eval_seqs if args.max_eval_seqs > 0 else None
    args.func(args)


if __name__ == '__main__':
    main()
