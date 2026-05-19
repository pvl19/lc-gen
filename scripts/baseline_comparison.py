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

EVAL_K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

NAIVE_METHODS = ('nn_mean', 'window_mean')


def _naive_predict(flux: torch.Tensor, js: torch.Tensor, k: int, C: int,
                   arange_C: torch.Tensor):
    """Return {'nn_mean', 'window_mean'} predictions for the given js."""
    pred_nn = 0.5 * (flux[js - k] + flux[js + k])
    fwd_idx = js.unsqueeze(1) - k - C + arange_C.unsqueeze(0)
    bwd_idx = js.unsqueeze(1) + k + arange_C.unsqueeze(0)
    fwd = flux[fwd_idx].mean(dim=1)
    bwd = flux[bwd_idx].mean(dim=1)
    pred_wm = 0.5 * (fwd + bwd)
    return {'nn_mean': pred_nn, 'window_mean': pred_wm}


# ----------------------------- fit_baselines --------------------------------

def cmd_fit_baselines(args):
    """Fit a per-k Gaussian sigma for each naive baseline on training residuals.

    These baselines are deterministic, so sigma_k is just the residual std. We
    estimate it on a held-out subset of training stars (disjoint from both eval
    and sanity-val) so it isn't oracle-fit to the comparison set. Writes
    baseline_sigmas.json which `eval` loads to compute NLL + coverage.
    """
    split = json.loads(Path(args.split_path).read_text())
    full_index = build_index([Path(p) for p in split['h5_paths']])
    eval_ids = set(split['eval_gaia_ids'])
    sanity_ids = set(split['sanity_gaia_ids'])
    fit_idx = [i for i, e in enumerate(full_index)
               if e['gaia_id'] not in eval_ids and e['gaia_id'] not in sanity_ids]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(fit_idx)
    if args.fit_stars:
        fit_idx = fit_idx[:args.fit_stars]
    print(f'[fit] {len(fit_idx)} training sequences for sigma calibration')

    device = torch.device(args.device)
    store = SequenceStore(full_index, use_metadata=False, device=device)
    C = int(args.C)
    arange_C = torch.arange(C, device=device, dtype=torch.long)

    stats = {m: {k: {'sum_se': 0.0, 'n': 0} for k in EVAL_K_GRID}
             for m in NAIVE_METHODS}

    for entry, flux, ferr, times, _ in store.iter_chunks(fit_idx, shuffle_chunks=False, shuffle_within=False):
        L = entry['length']
        for k in EVAL_K_GRID:
            j_min = k + C
            j_max = L - k - C
            if j_max <= j_min or k > args.K_max:
                continue
            n_full = j_max - j_min
            if args.n_targets_per_seq and args.n_targets_per_seq < n_full:
                picks = rng.choice(n_full, size=args.n_targets_per_seq, replace=False)
                picks.sort()
                js = torch.from_numpy(picks.astype(np.int64) + j_min).to(device)
            else:
                js = torch.arange(j_min, j_max, device=device, dtype=torch.long)
            target = flux[js]
            preds = _naive_predict(flux, js, k, C, arange_C)
            for m, pred in preds.items():
                err = (target - pred)
                stats[m][k]['sum_se'] += float((err ** 2).sum().item())
                stats[m][k]['n'] += int(target.numel())

    store.close()
    sigmas = {}
    for m, by_k in stats.items():
        sigmas[m] = {}
        for k, s in by_k.items():
            if s['n']:
                sigmas[m][k] = math.sqrt(s['sum_se'] / s['n'])
    out_path = Path(args.out_dir) / 'baseline_sigmas.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'sigmas': sigmas, 'n_fit_stars': len(fit_idx), 'C': C}, f, indent=2)
    print(f'[fit] wrote {out_path}')
    for m, by_k in sigmas.items():
        print(f'  {m}:  ' + '  '.join(f'k={k}: σ={s:.4f}' for k, s in sorted(by_k.items())))


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
    _d = ckpt if isinstance(ckpt, dict) else {}
    model = BiDirectionalMinGRU(
        hidden_size=64, direction='bi', mode='parallel',
        use_flow=has_flow, num_meta_features=num_meta_features,
        use_conv_channels=has_conv, meta_use_mask=_d.get('meta_use_mask', False),
        split_meta_encoders=_d.get('split_meta_encoders', False),
        instr_emb_dim=_d.get('instr_emb_dim', 16),
        adversarial_sector_head=_d.get('adversarial_sector_head', False),
        num_sectors=_d.get('num_sectors', 0),
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


NLL_CLIP = 50.0  # nats; caps individual blowups so a few overconfident points don't dominate the mean


class _StatAcc:
    """Running sums for MAE / RMSE / NLL / coverage per (method, k).

    Tracks two robust NLL summaries alongside the plain mean:
      - nll_clipped: per-point NLL is capped at NLL_CLIP before averaging
      - nll_median_seq: per-sequence mean NLL collected and median-ed
    Useful when a handful of clamped-sigma points blow up the unclipped mean.
    """

    def __init__(self):
        self.bins = {}

    def _bin(self, key):
        s = self.bins.get(key)
        if s is None:
            s = {'n': 0, 'sum_ae': 0.0, 'sum_se': 0.0,
                 'sum_nll': 0.0, 'sum_nll_clipped': 0.0, 'n_nll': 0,
                 'n_c68': 0, 'n_c95': 0, 'n_cov': 0,
                 'seqs': set(), 'seq_nll_sum': {}, 'seq_nll_n': {}}
            self.bins[key] = s
        return s

    @staticmethod
    def _record_seq_nll(s, seq_id, nll_sum: float, n: int):
        if seq_id is None or n == 0:
            return
        s['seq_nll_sum'][seq_id] = s['seq_nll_sum'].get(seq_id, 0.0) + nll_sum
        s['seq_nll_n'][seq_id] = s['seq_nll_n'].get(seq_id, 0) + n

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
        n = int(target.numel())
        s['n'] += n
        s['sum_ae'] += float(err.abs().sum().item())
        s['sum_se'] += float((err ** 2).sum().item())
        nll = 0.5 * (torch.log(2 * math.pi * sigma ** 2) + (err / sigma) ** 2)
        nll_sum = float(nll.sum().item())
        s['sum_nll'] += nll_sum
        s['sum_nll_clipped'] += float(nll.clamp(max=NLL_CLIP).sum().item())
        s['n_nll'] += n
        z = err / sigma
        s['n_c68'] += int((z.abs() <= 1.0).sum().item())
        s['n_c95'] += int((z.abs() <= 1.96).sum().item())
        s['n_cov'] += n
        if seq_id is not None:
            s['seqs'].add(seq_id)
            self._record_seq_nll(s, seq_id, nll_sum, n)

    def add_flow(self, method, k, target, log_prob, samples, seq_id=None):
        s = self._bin((method, k))
        nll = -log_prob
        nll_sum = float(nll.sum().item())
        n = int(target.numel())
        s['sum_nll'] += nll_sum
        s['sum_nll_clipped'] += float(nll.clamp(max=NLL_CLIP).sum().item())
        s['n_nll'] += n
        if seq_id is not None:
            self._record_seq_nll(s, seq_id, nll_sum, n)
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
            seq_means = []
            for sid in s['seq_nll_n']:
                sn = s['seq_nll_n'][sid]
                if sn:
                    seq_means.append(s['seq_nll_sum'][sid] / sn)
            nll_median_seq = float(np.median(seq_means)) if seq_means else float('nan')
            row = {
                'method': method,
                'k': k,
                'n_points': s['n'],
                'n_seqs': len(s['seqs']),
                'mae': s['sum_ae'] / n if s['n'] else float('nan'),
                'rmse': math.sqrt(s['sum_se'] / n) if s['n'] else float('nan'),
                'nll': (s['sum_nll'] / n_nll) if n_nll else float('nan'),
                'nll_clipped': (s['sum_nll_clipped'] / n_nll) if n_nll else float('nan'),
                'nll_median_seq': nll_median_seq,
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

    # Optional: per-k sigma for the naive baselines (enables NLL + coverage).
    baseline_sigmas = {}
    sigmas_path = Path(args.baseline_sigmas) if args.baseline_sigmas else None
    if sigmas_path and sigmas_path.exists():
        payload = json.loads(sigmas_path.read_text())
        for m, by_k in payload.get('sigmas', {}).items():
            baseline_sigmas[m] = {int(k): float(s) for k, s in by_k.items()}
        print(f'[eval] loaded naive-baseline sigmas from {sigmas_path}')
    elif sigmas_path:
        print(f'[eval] {sigmas_path} not found -- naive baselines will report MAE/RMSE only')

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

                # Naive baselines. If sigma_k is available, route through the
                # Gaussian accumulator to compute NLL + coverage.
                preds = _naive_predict(flux, js, k, C, arange_C)
                for m, pred in preds.items():
                    sigma_k = baseline_sigmas.get(m, {}).get(k)
                    if sigma_k is not None:
                        sigma_t = torch.full_like(pred, float(sigma_k))
                        acc.add_gaussian(m, k, target, pred, sigma_t, seq_id=seq_id)
                    else:
                        acc.add_point(m, k, target, pred, seq_id=seq_id)

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
    cols = ['method', 'k', 'n_points', 'n_seqs', 'mae', 'rmse',
            'nll', 'nll_clipped', 'nll_median_seq',
            'coverage68', 'coverage95']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'[eval] wrote {csv_path} ({len(rows)} rows)')
    by_method = {}
    for r in rows:
        by_method.setdefault(r['method'], []).append(r)
    for method, rs in sorted(by_method.items()):
        rs.sort(key=lambda x: x['k'])
        print(f'  {method}')
        print(f'    k     MAE      RMSE     NLL      NLLclip  NLLmed   cov68   cov95')
        for r in rs:
            def _fmt(v):
                return '   nan ' if (isinstance(v, float) and math.isnan(v)) else f'{v:7.4f}'
            c68 = '  nan' if math.isnan(r['coverage68']) else f'{r["coverage68"]:5.3f}'
            c95 = '  nan' if math.isnan(r['coverage95']) else f'{r["coverage95"]:5.3f}'
            print(f'    {r["k"]:<4d}  {_fmt(r["mae"])}  {_fmt(r["rmse"])}  {_fmt(r["nll"])}  '
                  f'{_fmt(r["nll_clipped"])}  {_fmt(r["nll_median_seq"])}  {c68}   {c95}')


# ----------------------------- plot -----------------------------------------

METHOD_STYLES = {
    'rnn_flow':     {'color': 'C0', 'marker': 'o', 'label': 'RNN + flow'},
    'mlp_gaussian': {'color': 'C1', 'marker': 's', 'label': 'MLP (Gaussian)'},
    'window_mean':  {'color': 'C2', 'marker': '^', 'label': 'window mean'},
    'nn_mean':      {'color': 'C3', 'marker': 'v', 'label': 'NN mean'},
}


def _read_summary(csv_path: Path):
    rows = []
    with open(csv_path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            parts = line.strip().split(',')
            if not parts or parts == ['']:
                continue
            row = dict(zip(header, parts))
            row['k'] = int(row['k'])
            for col in ('mae', 'rmse', 'nll', 'nll_clipped', 'nll_median_seq',
                        'coverage68', 'coverage95'):
                if col in row:
                    row[col] = float(row[col])
            rows.append(row)
    return rows


def _grouped(rows, metric: str):
    """Return {method: ([k...], [val...])} keeping only non-NaN entries."""
    out = {}
    for r in rows:
        v = r[metric]
        if math.isnan(v):
            continue
        out.setdefault(r['method'], ([], []))
        out[r['method']][0].append(r['k'])
        out[r['method']][1].append(v)
    for m in out:
        ks, vs = out[m]
        order = sorted(range(len(ks)), key=lambda i: ks[i])
        out[m] = ([ks[i] for i in order], [vs[i] for i in order])
    return out


# ----------------------------- linear probe ---------------------------------

PROBE_K_GRID = [16, 64, 256, 512, 720]


def _collect_probe_features(rnn, store, indices, k_grid, n_targets, C, rng, device, label):
    """For each sequence, run RNN once, then for each k pull (h_fwd[j-k], h_bwd[j+k],
    t_enc[j], ferr[j]) for a random subsample of valid j. Returns dict {k: (X, y)}.
    """
    bins = {k: {'X': [], 'y': []} for k in k_grid}
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for entry, flux, ferr, times, meta in store.iter_chunks(
                indices, shuffle_chunks=False, shuffle_within=False):
            L = entry['length']
            cache = _rnn_forward_cache(rnn, flux, ferr, times, meta)
            h_fwd = cache['h_fwd'][0]   # (L, Hf)
            h_bwd = cache['h_bwd'][0]   # (L, Hb)
            t_enc = cache['t_enc'][0]   # (L, Te)
            for k in k_grid:
                j_min = k + C
                j_max = L - k - C
                if j_max <= j_min:
                    continue
                n_full = j_max - j_min
                if n_targets and n_targets < n_full:
                    picks = rng.choice(n_full, size=n_targets, replace=False)
                    picks.sort()
                    js = torch.from_numpy(picks.astype(np.int64) + j_min).to(device)
                else:
                    js = torch.arange(j_min, j_max, device=device, dtype=torch.long)
                feat = torch.cat([
                    h_fwd[js - k], h_bwd[js + k], t_enc[js],
                    ferr[js].unsqueeze(-1),
                ], dim=-1).cpu().numpy().astype(np.float32)
                tgt = flux[js].cpu().numpy().astype(np.float32)
                bins[k]['X'].append(feat)
                bins[k]['y'].append(tgt)
            n += 1
            if n % 100 == 0:
                dt = time.time() - t0
                rate = n / dt if dt else 0.0
                print(f'  [{label}] {n}/{len(indices)} seqs ({rate:.2f} seq/s)')
    out = {}
    for k, v in bins.items():
        if v['X']:
            out[k] = (np.concatenate(v['X']), np.concatenate(v['y']))
    return out


def _fit_ridge(X, y, lam):
    """Closed-form ridge: w = (XᵀX + λI)⁻¹ Xᵀy with bias column."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d, dtype=Xb.dtype)
    A[-1, -1] -= lam  # don't regularize bias
    b = Xb.T @ y
    w = np.linalg.solve(A, b)
    return w


def _apply_ridge(X, w):
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    return Xb @ w


def cmd_linear_probe(args):
    """Linear probe: fit ridge regression on RNN latents → flux(j) per k.

    Diagnoses whether the RNN's hidden state still contains information about
    flux(j) at large k. If linear-probe MAE is close to the local MLP's MAE,
    the latent has the info and the flow head is the bottleneck. If much worse,
    bigger hidden / longer training horizon may help.
    """
    split = json.loads(Path(args.split_path).read_text())
    full_index = build_index([Path(p) for p in split['h5_paths']])
    eval_ids = set(split['eval_gaia_ids'])
    sanity_ids = set(split['sanity_gaia_ids'])
    train_idx = [i for i, e in enumerate(full_index)
                 if e['gaia_id'] not in eval_ids and e['gaia_id'] not in sanity_ids]
    eval_idx = [i for i, e in enumerate(full_index) if e['gaia_id'] in eval_ids]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_idx)
    if args.train_stars:
        train_idx = train_idx[:args.train_stars]
    if args.eval_stars:
        rng.shuffle(eval_idx)
        eval_idx = eval_idx[:args.eval_stars]
    print(f'[probe] train={len(train_idx)}  eval={len(eval_idx)}  k_grid={PROBE_K_GRID}')

    device = torch.device(args.device)
    store = SequenceStore(full_index, use_metadata=True, device=device)
    rnn = _load_rnn_model(args.rnn_path, device, num_meta_features=len(METADATA_FEATURES))

    print('[probe] collecting train features...')
    train_bins = _collect_probe_features(
        rnn, store, train_idx, PROBE_K_GRID, args.n_targets_per_seq, args.C, rng, device, 'train')
    print('[probe] collecting eval features...')
    eval_bins = _collect_probe_features(
        rnn, store, eval_idx, PROBE_K_GRID, args.n_targets_per_seq, args.C, rng, device, 'eval')
    store.close()

    rows = []
    print(f'{"k":>5}  {"d":>4}  {"n_tr":>7}  {"n_ev":>7}  {"MAE":>8}  {"RMSE":>8}')
    for k in PROBE_K_GRID:
        if k not in train_bins or k not in eval_bins:
            continue
        Xtr, ytr = train_bins[k]
        Xev, yev = eval_bins[k]
        w = _fit_ridge(Xtr, ytr, lam=args.ridge_lambda)
        pred = _apply_ridge(Xev, w)
        err = yev - pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        print(f'{k:>5d}  {Xtr.shape[1]:>4d}  {len(ytr):>7d}  {len(yev):>7d}  {mae:>8.4f}  {rmse:>8.4f}')
        rows.append({'method': 'rnn_linear_probe', 'k': k,
                     'n_train': int(len(ytr)), 'n_eval': int(len(yev)),
                     'mae': mae, 'rmse': rmse,
                     'feat_dim': int(Xtr.shape[1])})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'linear_probe.csv'
    cols = ['method', 'k', 'feat_dim', 'n_train', 'n_eval', 'mae', 'rmse']
    with open(out_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'[probe] wrote {out_path}')


def cmd_plot(args):
    import matplotlib.pyplot as plt

    csv_path = Path(args.summary_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found -- run `eval` first')
    rows = _read_summary(csv_path)
    exclude = set(args.exclude_methods or [])
    if exclude:
        rows = [r for r in rows if r['method'] not in exclude]
        print(f'[plot] excluded methods: {sorted(exclude)}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ('nll',            'NLL (nats/point)',          'nll_vs_k.png'),
        ('nll_clipped',    'NLL clipped @ 50 nats',     'nll_clipped_vs_k.png'),
        ('nll_median_seq', 'median per-sequence NLL',   'nll_median_seq_vs_k.png'),
        ('mae',            'MAE',                       'mae_vs_k.png'),
        ('rmse',           'RMSE',                      'rmse_vs_k.png'),
    ]
    for metric, ylabel, fname in panels:
        grouped = _grouped(rows, metric)
        if not grouped:
            print(f'[plot] no rows for {metric} -- skipping')
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for method, (ks, vs) in sorted(grouped.items()):
            style = METHOD_STYLES.get(method, {'color': None, 'marker': 'o', 'label': method})
            ax.plot(ks, vs, marker=style['marker'], color=style['color'],
                    linewidth=1.5, markersize=6, label=style['label'])
        ax.set_xscale('log', base=2)
        ax.set_xlabel('prediction offset k')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs. k on eval-10%')
        ax.grid(True, which='both', linestyle=':', alpha=0.4)
        ax.legend(loc='best', fontsize=9)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[plot] wrote {path}')


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

    fp = sub.add_parser('fit_baselines',
                        help='Fit per-k Gaussian sigma for naive baselines on training residuals.')
    fp.add_argument('--split-path', default='output/baseline_comparison/split.json')
    fp.add_argument('--out-dir', default='output/baseline_comparison')
    fp.add_argument('--device', default='cpu')
    fp.add_argument('--seed', type=int, default=0)
    fp.add_argument('--K-max', type=int, default=720)
    fp.add_argument('--C', type=int, default=32)
    fp.add_argument('--fit-stars', type=int, default=500,
                    help='Subsample of training sequences for sigma calibration (0 = all).')
    fp.add_argument('--n-targets-per-seq', type=int, default=256)
    fp.set_defaults(func=cmd_fit_baselines)

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
    ep.add_argument('--baseline-sigmas',
                    default='output/baseline_comparison/baseline_sigmas.json',
                    help='Path to per-k Gaussian sigma for naive baselines. '
                         'Pass empty string to skip NLL/coverage for them.')
    ep.set_defaults(func=cmd_eval)

    pp = sub.add_parser('plot', help='Plot NLL/MAE/RMSE vs k from summary.csv.')
    pp.add_argument('--summary-csv', default='output/baseline_comparison/summary.csv')
    pp.add_argument('--out-dir', default='output/baseline_comparison')
    pp.add_argument('--exclude-methods', nargs='*', default=['nn_mean'],
                    help='Methods to omit from the plots (kept in summary.csv).')
    pp.set_defaults(func=cmd_plot)

    lp = sub.add_parser('linear_probe',
                        help='Linear probe on RNN latents: ridge regression → flux(j) per k.')
    lp.add_argument('--split-path', default='output/baseline_comparison/split.json')
    lp.add_argument('--out-dir', default='output/baseline_comparison')
    lp.add_argument('--rnn-path', required=True)
    lp.add_argument('--device', default='cpu')
    lp.add_argument('--seed', type=int, default=0)
    lp.add_argument('--C', type=int, default=32,
                    help='Edge guard so j±k stays away from the boundary.')
    lp.add_argument('--train-stars', type=int, default=200,
                    help='Training sequences used to fit the linear head (0 = all).')
    lp.add_argument('--eval-stars', type=int, default=500,
                    help='Eval sequences scored (0 = all eval-10%%).')
    lp.add_argument('--n-targets-per-seq', type=int, default=256)
    lp.add_argument('--ridge-lambda', type=float, default=1.0)
    lp.set_defaults(func=cmd_linear_probe)

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
