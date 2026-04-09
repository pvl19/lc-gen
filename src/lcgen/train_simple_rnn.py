"""Minimal training script for the single-cell SimpleMinGRU model.

This script is intentionally small: it loads an HDF5 timeseries file, creates
a simple dataloader, trains the single-cell model for a few epochs, and saves
a checkpoint. It supports the optional zuko flow if available; otherwise it
falls back to Gaussian negative-log-likelihood (or MSE for simplicity).
"""
import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import datetime

import sys
from pathlib import Path as _P
# Ensure local 'src' is on sys.path so we can import lcgen
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from lcgen.models.simple_min_gru import SimpleMinGRU, BiDirectionalMinGRU
from lcgen.utils.trunc_data import extract_data
from lcgen.utils.loss import recon_loss, bounded_horizon_future_nll
from lcgen.utils.run_log import log_run_args
from lcgen.models.TimeSeriesDataset import TimeSeriesDataset, LazyH5Dataset, collate_fn


def setup_ddp():
    """Initialize DDP process group if running under torchrun, else return rank=0."""
    if 'LOCAL_RANK' not in os.environ:
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    return rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def worker_init_fn(worker_id):
    """Reseed each DataLoader worker so block masks are not correlated across workers."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def validate(model, val_loader, device, args):
    """Compute validation loss without updating model parameters.

    If val_k_values is provided, compute loss at each fixed k value and average them.
    Otherwise, use random k sampling like training.
    """
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            # batch is a tuple: (flux, flux_err, time, mask, metadata, conv_data)
            flux, flux_err, times, mask, metadata, conv_data = batch
            flux = flux.to(device)
            flux_err = flux_err.to(device)
            times = times.to(device)
            mask = mask.to(device)
            metadata = metadata.to(device) if args.use_metadata else None

            # Move conv_data to device if present
            if conv_data is not None:
                conv_data = {k: v.to(device) for k, v in conv_data.items()}

            # Build input channels [flux, flux_err] -> (B, L, 2)
            x_in = torch.stack([flux, flux_err], dim=-1)
            t_in = torch.stack([times], dim=-1)

            # Request hidden states for multi-step supervision
            out = model(x_in, t_in, mask=mask, metadata=metadata, conv_data=conv_data, return_states=True)

            # Extract forward/backward hidden states and time encodings
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')
            t_enc = out.get('t_enc')

            # Compute bounded-horizon averaged NLL over future predictions
            if hasattr(args, 'val_k_values') and args.val_k_values:
                # Use fixed k values for validation
                k_losses = []
                for k_val in args.val_k_values:
                    loss, stats, per_k_mean = bounded_horizon_future_nll(
                        h_fwd, h_bwd, t_enc, model, flux, flux_err,
                        mask=mask, metadata=metadata, K=args.K, k_spacing=args.k_spacing, fixed_k=k_val
                    )
                    k_losses.append(loss.item())
                # Average across k values
                batch_loss = sum(k_losses) / len(k_losses)
            else:
                # Use random k sampling (same as training)
                loss, stats, per_k_mean = bounded_horizon_future_nll(
                    h_fwd, h_bwd, t_enc, model, flux, flux_err,
                    mask=mask, metadata=metadata, K=args.K, k_spacing=args.k_spacing
                )
                batch_loss = loss.item()

            total_loss += batch_loss
            n += 1

    avg_val_loss = total_loss / max(1, n)
    return avg_val_loss

def _save_rolling_checkpoint(
    model, optimizer, epoch,
    previous_train_losses, train_epoch_losses,
    previous_val_losses, val_epoch_losses,
    k_stats_per_epoch, output_path, persistent_dir,
):
    """Save a rolling checkpoint and losses after each epoch, then optionally
    copy both files to a persistent directory (e.g. the project home dir on
    Bridges-2) so progress is safe even if the SLURM job is killed mid-run."""
    import shutil

    all_train = previous_train_losses + train_epoch_losses
    all_val   = previous_val_losses   + val_epoch_losses

    # Write checkpoint_latest.pt  (always overwritten – no storage bloat)
    ckpt_path = output_path / 'checkpoint_latest.pt'
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'num_meta_features':    model.num_meta_features,
    }, ckpt_path)

    # Write losses_per_epoch.npz
    losses_path = output_path / 'losses_per_epoch.npz'
    save_dict = {'train_loss': np.array(all_train, dtype=np.float32)}
    if all_val:
        save_dict['val_loss'] = np.array(all_val, dtype=np.float32)
    if k_stats_per_epoch:
        save_dict['k_mean'] = np.array([s['mean'] for s in k_stats_per_epoch], dtype=np.float32)
        save_dict['k_min']  = np.array([s['min']  for s in k_stats_per_epoch], dtype=np.int32)
        save_dict['k_max']  = np.array([s['max']  for s in k_stats_per_epoch], dtype=np.int32)
        save_dict['k_std']  = np.array([s['std']  for s in k_stats_per_epoch], dtype=np.float32)
    np.savez_compressed(losses_path, **save_dict)

    print(f'  → Saved rolling checkpoint (epoch {epoch}): {ckpt_path}')

    # Copy to persistent storage so the files survive a job timeout
    if persistent_dir is not None:
        shutil.copy2(ckpt_path,    persistent_dir / 'model.pt')
        shutil.copy2(losses_path,  persistent_dir / 'losses_per_epoch.npz')
        args_src = output_path / 'train_args.json'
        if args_src.exists():
            shutil.copy2(args_src, persistent_dir / 'train_args.json')
        print(f'  → Copied checkpoint to {persistent_dir}')


def train(args):
    rank, world_size, device = setup_ddp()
    is_main = (rank == 0)
    ddp_active = (world_size > 1)

    if is_main:
        print(f"World size: {world_size}")
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(device.index)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.1f} GB")

    # Build dataset — one LazyH5Dataset per input file, concatenated
    seed_offset = rank  # different mask seeds per rank
    datasets = [
        LazyH5Dataset(
            path, args.random_seed + seed_offset,
            args.min_size, args.max_size, args.mask_portion,
            num_samples=args.num_samples,
            use_metadata=args.use_metadata,
            metadata_features=None,
        )
        for path in args.input
    ]
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    num_meta_features = datasets[0].num_meta_features if args.use_metadata else 0

    if is_main:
        print(f'Total samples: {len(ds)} across {len(datasets)} file(s)')

    # Split into train / val
    val_size = int(len(ds) * args.val_split) if args.val_split > 0 else 0
    train_size = len(ds) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(ds, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(args.random_seed))
        if is_main:
            print(f'Split: {train_size} train, {val_size} val')
    else:
        train_ds, val_ds = ds, None
        if is_main:
            print('No validation split - using all data for training')

    # Samplers — DDP uses DistributedSampler; single-GPU uses default shuffle
    if ddp_active:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) \
                        if val_ds is not None else None
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler   = None
        shuffle_train = True

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle_train,
                              sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn) \
                 if val_ds is not None else None

    # Convolutional channel configuration
    conv_config = None
    if args.use_conv_channels:
        # Choose encoder type based on your priorities:
        # 'unet' (default): Richer multi-scale patterns with skip connections - RECOMMENDED for science
        # 'lightweight': Faster but simpler - good for rapid iteration
        encoder_type = args.conv_encoder_type

        if encoder_type == 'lightweight':
            conv_config = {
                'encoder_type': 'lightweight',
                'hidden_channels': 16,
                'num_layers': 3,
                'activation': 'gelu'
            }
            print(f"Using LIGHTWEIGHT conv encoder (faster, simpler)")
        else:  # unet
            conv_config = {
                'encoder_type': 'unet',
                'input_length': 16000,
                'encoder_dims': [4, 8, 16, 32],
                'num_layers': 4,
                'activation': 'gelu'
            }
            print(f"Using UNET conv encoder (slower but richer multi-scale features)")

    raw_model = BiDirectionalMinGRU(hidden_size=args.hidden_size, direction=args.direction, mode=args.mode, use_flow=args.use_flow, num_meta_features=num_meta_features, use_conv_channels=args.use_conv_channels, conv_config=conv_config).to(device)

    # Compile before DDP so the compiler sees the full model graph.
    # dynamic=True avoids recompilation when padded sequence length varies across batches.
    # torch.compile requires CUDA capability >= 7.5 (inductor backend); skip on V100 (sm_70).
    cuda_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if cuda_cap >= (7, 5):
        raw_model = torch.compile(raw_model, dynamic=True)
        if is_main:
            print(f'torch.compile enabled (CUDA capability {cuda_cap[0]}.{cuda_cap[1]})')
    else:
        if is_main:
            print(f'torch.compile skipped (CUDA capability {cuda_cap[0]}.{cuda_cap[1]} < 7.5)')

    if ddp_active:
        model = DDP(raw_model, device_ids=[device.index])
        model._set_static_graph()
    else:
        model = raw_model

    if is_main:
        if hasattr(raw_model, 'time_scale'):
            print('time_scale present; requires_grad =', raw_model.time_scale.requires_grad)
        if hasattr(raw_model, 'flow') and raw_model.flow is not None:
            print('Flow head enabled and present on model; using zuko-based flow head.')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Handle checkpoint resuming
    start_epoch = 0
    previous_train_losses = []
    previous_val_losses = []

    if args.resume_from:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {args.resume_from}")
        print(f"{'='*60}")

        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model state from checkpoint")

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Loaded optimizer state from checkpoint")

        # Load previous losses from the checkpoint directory (REQUIRED for resume)
        checkpoint_dir = checkpoint_path.parent
        losses_path = checkpoint_dir / 'losses_per_epoch.npz'
        if not losses_path.exists():
            raise FileNotFoundError(
                f"losses_per_epoch.npz not found in {checkpoint_dir}\n"
                f"When resuming training, both files are required:\n"
                f"  1. {checkpoint_path.name} (checkpoint)\n"
                f"  2. losses_per_epoch.npz (loss history)\n"
                f"Both files must be in the same directory."
            )

        losses_data = np.load(losses_path)
        previous_train_losses = list(losses_data['train_loss'])
        if 'val_loss' in losses_data:
            previous_val_losses = list(losses_data['val_loss'])
        start_epoch = len(previous_train_losses)
        print(f"✓ Loaded previous losses: {start_epoch} epochs already completed")
        print(f"  Last train loss: {previous_train_losses[-1]:.6f}")
        if previous_val_losses:
            print(f"  Last val loss: {previous_val_losses[-1]:.6f}")

        # Check if we've already reached the target epoch count
        if start_epoch >= args.epochs:
            print(f"\n⚠ Model has already been trained for {start_epoch} epochs (target: {args.epochs})")
            print(f"  No additional training needed. Set --epochs to a higher value to continue training.")
            return

        print(f"\nResuming training from epoch {start_epoch + 1} to epoch {args.epochs}")
        print(f"{'='*60}\n")

    # Calculate remaining epochs for scheduler
    remaining_epochs = args.epochs - start_epoch
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=remaining_epochs, pct_start=0.3, div_factor=10.0, final_div_factor=1000.0)

    if remaining_epochs <= 0:
        print('No epochs requested, exiting.')
        return

    # trackers for epoch-level logging (will be concatenated with previous losses when saving)
    train_epoch_losses = []     # NEW training loss per epoch (for this session)
    val_epoch_losses = []       # NEW validation loss per epoch (for this session)
    k_stats_per_epoch = []      # NEW k-value statistics per epoch (mean, min, max)

    # Set up output path early so per-epoch checkpoints can be written inside the loop
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up persistent checkpoint dir for mid-job copies
    persistent_dir = Path(args.checkpoint_copy_dir) if args.checkpoint_copy_dir else None
    if persistent_dir is not None:
        persistent_dir.mkdir(parents=True, exist_ok=True)

    # early stopping tracking
    # If resuming, initialize best_val_loss from previous validation losses
    if previous_val_losses:
        best_val_loss = min(previous_val_losses)
        print(f"Initialized best_val_loss from previous training: {best_val_loss:.6f}")
    else:
        best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    stopped_early = False

    import time as time_module
    for epoch in range(start_epoch, args.epochs):
        if ddp_active:
            train_sampler.set_epoch(epoch)
        epoch_start = time_module.time()
        model.train()
        total_loss = 0.0
        n = 0
        epoch_k_values = []  # Collect sampled k values for this epoch

        # Timing accumulators for profiling
        time_data_load = 0.0
        time_forward = 0.0
        time_loss = 0.0
        time_backward = 0.0
        time_optimizer = 0.0

        for batch in train_loader:
            batch_start = time_module.time()

            optimizer.zero_grad()
            # batch is a tuple: (flux, flux_err, time, mask, metadata, conv_data)
            flux, flux_err, times, mask, metadata, conv_data = batch
            flux = flux.to(device)
            flux_err = flux_err.to(device)
            times = times.to(device)
            mask = mask.to(device)
            metadata = metadata.to(device) if args.use_metadata else None

            # Move conv_data to device if present
            if conv_data is not None:
                conv_data = {k: v.to(device) for k, v in conv_data.items()}

            # Build input channels [flux, flux_err] -> (B, L, 2)
            x_in = torch.stack([flux, flux_err], dim=-1)
            t_in = torch.stack([times], dim=-1)

            time_data_load += time_module.time() - batch_start

            # Request hidden states for multi-step supervision
            # Pass mask so the model zeros out masked flux/flux_err in RNN input
            forward_start = time_module.time()
            out = model(x_in, t_in, mask=mask, metadata=metadata, conv_data=conv_data, return_states=True)
            recon = out['reconstructed']  # (B, L, 1)

            # Extract forward/backward hidden states and time encodings
            h_fwd = out.get('h_fwd_tensor')
            h_bwd = out.get('h_bwd_tensor')
            t_enc = out.get('t_enc')
            time_forward += time_module.time() - forward_start

            # Compute bounded-horizon averaged NLL over future predictions
            # K chosen as 32 by default (can tune)
            # Pass the model so the loss uses the same head_norm / time-conditioning
            # that `model.forward` applies during inference. Also pass h_bwd so
            # bidirectional models can form the same fused head input.
            # Pass mask so loss is only computed on valid (unmasked) predictions.
            # Pass metadata so the head can use stellar properties for predictions.
            loss_start = time_module.time()
            loss, stats, per_k_mean = bounded_horizon_future_nll(h_fwd, h_bwd, t_enc, model, flux, flux_err, mask=mask, metadata=metadata, K=args.K, k_spacing=args.k_spacing)

            # Track sampled k value for this batch
            if 'sampled_k' in stats:
                epoch_k_values.append(stats['sampled_k'])
            time_loss += time_module.time() - loss_start

            backward_start = time_module.time()
            loss.backward()
            time_backward += time_module.time() - backward_start

            optimizer_start = time_module.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            time_optimizer += time_module.time() - optimizer_start

            total_loss += loss.item()
            n += 1

        avg = total_loss / max(1, n)
        # All-reduce training loss across ranks so rank 0 logs the global average
        if ddp_active:
            t = torch.tensor(avg, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg = t.item()
        epoch_time = time_module.time() - epoch_start
        if is_main:
            train_epoch_losses.append(avg)

        # Compute training time breakdown
        time_train_total = time_data_load + time_forward + time_loss + time_backward + time_optimizer

        # Compute k-value statistics for this epoch
        if epoch_k_values:
            k_mean = np.mean(epoch_k_values)
            k_min = np.min(epoch_k_values)
            k_max = np.max(epoch_k_values)
            k_std = np.std(epoch_k_values)
            k_stats_per_epoch.append({
                'mean': k_mean,
                'min': k_min,
                'max': k_max,
                'std': k_std,
            })

        # Run validation if validation set exists
        if val_loader is not None:
            val_start = time_module.time()
            val_loss = validate(model, val_loader, device, args)
            if ddp_active:
                t = torch.tensor(val_loss, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_loss = t.item()
            time_val = time_module.time() - val_start
            if is_main:
                val_epoch_losses.append(val_loss)
                print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {avg:.6f} - Val Loss: {val_loss:.6f} - Time: {epoch_time:.1f}s')
                print(f'  Timing breakdown: Data={time_data_load:.1f}s Forward={time_forward:.1f}s Loss={time_loss:.1f}s Backward={time_backward:.1f}s Optimizer={time_optimizer:.1f}s Val={time_val:.1f}s')

                # Early stopping check
                if args.patience > 0:
                    if val_loss < (best_val_loss - args.min_delta):
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        best_model_state = {
                            'model_state_dict': raw_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch + 1
                        }
                        print(f'  → New best validation loss: {best_val_loss:.6f}')
                    else:
                        epochs_without_improvement += 1
                        print(f'  → No improvement for {epochs_without_improvement} epoch(s) (best: {best_val_loss:.6f})')

                        if epochs_without_improvement >= args.patience:
                            print(f'\nEarly stopping triggered after {epoch+1} epochs (patience={args.patience})')
                            print(f'Best validation loss: {best_val_loss:.6f} at epoch {epoch+1-args.patience}')
                            stopped_early = True
                            _save_rolling_checkpoint(
                                raw_model, optimizer, epoch + 1,
                                previous_train_losses, train_epoch_losses,
                                previous_val_losses, val_epoch_losses,
                                k_stats_per_epoch, output_path, persistent_dir,
                            )
            # Broadcast early-stop decision from rank 0 to all ranks
            if ddp_active and args.patience > 0:
                stop_flag = torch.tensor(int(stopped_early), device=device)
                dist.broadcast(stop_flag, src=0)
                stopped_early = bool(stop_flag.item())
            if stopped_early:
                break
        else:
            if is_main:
                print(f'Epoch {epoch+1}/{args.epochs} - Loss: {avg:.6f} - Time: {epoch_time:.1f}s')
                print(f'  Timing breakdown: Data={time_data_load:.1f}s Forward={time_forward:.1f}s Loss={time_loss:.1f}s Backward={time_backward:.1f}s Optimizer={time_optimizer:.1f}s')

        # ---- Per-epoch rolling checkpoint (rank 0 only) ----
        if is_main and not stopped_early and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save_rolling_checkpoint(
                raw_model, optimizer, epoch + 1,
                previous_train_losses, train_epoch_losses,
                previous_val_losses, val_epoch_losses,
                k_stats_per_epoch, output_path, persistent_dir,
            )

    if is_main:
        # Restore best model if early stopping was used
        if best_model_state is not None:
            raw_model.load_state_dict(best_model_state['model_state_dict'])
            optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
            best_epoch = best_model_state['epoch']
            print(f'\nRestored best model from epoch {best_epoch}')

        # Concatenate losses with previous training session
        all_train_losses = previous_train_losses + train_epoch_losses
        all_val_losses = previous_val_losses + val_epoch_losses

        # Save final/best model with optimizer state for potential resumption
        model_path = output_path / args.output_name
        checkpoint_dict = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': len(all_train_losses),
            'num_meta_features': raw_model.num_meta_features,
        }
        torch.save(checkpoint_dict, model_path)
        if stopped_early:
            print(f'Saved best model (epoch {best_epoch}) to {model_path}')
        else:
            print(f'Saved final model (epoch {len(all_train_losses)}) to {model_path}')

    # Save epoch-level losses — rank 0 only
    if is_main:
        try:
            losses_path = output_path / 'losses_per_epoch.npz'
            save_dict = {'train_loss': np.array(all_train_losses, dtype=np.float32)}
            if val_loader is not None and len(all_val_losses) > 0:
                save_dict['val_loss'] = np.array(all_val_losses, dtype=np.float32)
            if k_stats_per_epoch:
                save_dict['k_mean'] = np.array([s['mean'] for s in k_stats_per_epoch], dtype=np.float32)
                save_dict['k_min']  = np.array([s['min']  for s in k_stats_per_epoch], dtype=np.int32)
                save_dict['k_max']  = np.array([s['max']  for s in k_stats_per_epoch], dtype=np.int32)
                save_dict['k_std']  = np.array([s['std']  for s in k_stats_per_epoch], dtype=np.float32)
            if stopped_early:
                save_dict['stopped_early'] = True
                save_dict['best_epoch'] = best_epoch if best_model_state else len(all_train_losses) - epochs_without_improvement
                save_dict['best_val_loss'] = best_val_loss
            np.savez_compressed(losses_path, **save_dict)
            print('Saved epoch losses to', losses_path)
            if previous_train_losses:
                print(f'  Combined {len(previous_train_losses)} previous + {len(train_epoch_losses)} new epochs = {len(all_train_losses)} total')
        except Exception as e:
            print('Warning: failed to save epoch losses:', e)

    cleanup_ddp()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, nargs='+', default=['data/timeseries_x.h5'],
                   help='One or more HDF5 input files (e.g. --input file1.h5 file2.h5)')
    p.add_argument('--direction', type=str, default='forward', choices=['forward', 'backward', 'bi'])
    p.add_argument('--random_seed', type=int, default=19)
    p.add_argument('--num_samples', type=int, default=None)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--output_dir', type=str, default='output/simple_rnn')
    p.add_argument('--output_name', type=str, default='simple_min_gru_default_output.pt')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--use_flow', action='store_true')
    p.add_argument('--min_size', type=int, default=2)
    p.add_argument('--max_size', type=int, default=100)
    p.add_argument('--mask_portion', type=float, default=0.2)
    p.add_argument('--mock_sinusoid', action='store_true')
    p.add_argument('--mock_noise', type=float, default=0.1)
    p.add_argument('--mode', type=str, default='sequential', choices=['sequential', 'parallel'])
    p.add_argument('--K', type=int, default=128)
    p.add_argument('--k_spacing', type=str, default='dense', choices=['dense', 'log'],
                   help="How to sample k offset value per batch: 'dense' (uniform 1..K), 'log' (log-uniform 1..K)")
    p.add_argument('--use_metadata', action='store_true',
                   help='Use stellar metadata (G_0, BP_0, RP_0, parallax + uncertainties) as model input')
    p.add_argument('--use_conv_channels', action='store_true',
                   help='Use convolutional encoders for power spectrum, f-statistic, and ACF as additional model input')
    p.add_argument('--conv_encoder_type', type=str, default='unet', choices=['lightweight', 'unet'],
                   help="Conv encoder architecture: 'unet' (default, richer but slower) or 'lightweight' (faster but simpler)")
    p.add_argument('--val_split', type=float, default=0.1,
                   help='Fraction of data to use for validation (default: 0.1 for 10%)')
    p.add_argument('--val_k_values', type=str, default='1,2,4,8',
                   help='Comma-separated list of fixed k values to use for validation loss (default: 1,2,4,8). This standardizes validation metrics across runs.')
    p.add_argument('--patience', type=int, default=0,
                   help='Early stopping patience: stop if validation loss does not improve for this many epochs (0 = disabled)')
    p.add_argument('--min_delta', type=float, default=0.0,
                   help='Minimum change in validation loss to qualify as improvement for early stopping')
    p.add_argument('--resume_from', type=str, default=None,
                   help='Path to checkpoint .pt file to resume training from. Will load model state, optimizer state, and previous losses.')
    p.add_argument('--save_every', type=int, default=1,
                   help='Save a rolling checkpoint every N epochs (default: 1). Set to 0 to disable mid-run saves.')
    p.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers per GPU process. Set to '
                             '(cpus_per_task - nproc_per_node) // nproc_per_node.')
    p.add_argument('--checkpoint_copy_dir', type=str, default=None,
                   help='If set, copy checkpoint_latest.pt and losses_per_epoch.npz here after every save. '
                        'Use a path on persistent storage (e.g. $HOME/lcgen/checkpoints/resume) so progress '
                        'survives a SLURM job timeout.')
    return p.parse_args()


if __name__ == '__main__':
    import sys
    args = parse_args()

    # Parse val_k_values from comma-separated string to list of integers
    if hasattr(args, 'val_k_values') and args.val_k_values:
        args.val_k_values = [int(k.strip()) for k in args.val_k_values.split(',')]
    else:
        args.val_k_values = None

    # Create output directory and save training arguments
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_run_args(args, log_file=str(output_path / 'train_args.json'))

    start = datetime.datetime.now()
    try:
        print("Run arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        train(args)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    end = datetime.datetime.now()
    print('Total training time:', end - start)
