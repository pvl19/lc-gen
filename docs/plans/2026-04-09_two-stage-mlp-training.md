# Two-Stage MLP Training & Linear Encoder for Age Inference

**Date:** 2026-04-09
**Status:** Implemented

## Context

The MLP encoder for age inference (r=0.18–0.25) dramatically underperforms PCA (r=0.53 at 8D). The core issue is **moving-target instability**: the encoder shifts every gradient step, so the flow chases a constantly changing distribution. This is analogous to GAN training instability — the flow's density estimate becomes stale as the encoder moves.

Two-stage training eliminates this by decoupling the encoder and flow training phases. A linear encoder baseline isolates whether the auxiliary loss helps at all vs PCA.

**Baseline:** PCA-8D, r=0.529, MAE=0.493 dex

## Training stages

### `joint` (default, original behavior)
All parameters trained together. NLL + aux L1 + variance reg.

### `two_stage`
1. **Stage 1 — Encoder pre-training** (`encoder_pretrain_epochs`):
   - Flow frozen (`requires_grad_(False)`)
   - Only encoder + aux_age_head parameters optimized
   - Loss = aux L1 only + variance reg (no NLL)
   - Own optimizer and LR scheduler

2. **Stage 2 — Flow training** (remaining epochs):
   - Encoder + aux_head frozen
   - Only flow parameters optimized
   - Loss = NLL only
   - Own optimizer and LR scheduler

Rationale: stage 1 gives the encoder a stable, age-informative representation. Stage 2 trains the flow on a fixed bottleneck distribution — no moving target.

### `three_stage`
Same as two_stage, plus:

3. **Stage 3 — Joint fine-tuning** (`joint_finetune_epochs`):
   - All parameters unfrozen
   - Full loss (NLL + aux + var_reg)
   - Separate LR groups: encoder at `lr * 0.01`, flow at `lr`
   - Lower encoder LR prevents the moving-target problem from returning

## Linear encoder

`--encoder_type linear` uses `AgePredictorMLP` with `mlp_hidden=[]`, which reduces the encoder to:

```
Linear(input_dim, bottleneck_dim) → LayerNorm
```

This is a learned linear projection directly comparable to PCA, but trained with the auxiliary age loss. It answers: does the aux loss find better projections than PCA's variance-optimal ones?

## `loss_mode` parameter

`AgePredictorMLP.forward` accepts `loss_mode`:
- `'full'` — NLL + aux L1 + variance reg (joint training)
- `'aux_only'` — aux L1 + variance reg (stage 1)
- `'nll_only'` — NLL only (stage 2)

## Implementation

### `_run_stage` helper (inside `train_single_fold`)
Extracted the epoch loop into a closure that accepts: params to optimize, number of epochs, LR, loss_mode, and label. Updates shared `train_losses`, `val_losses`, `best_loss`, `best_state` via nonlocal. Best model selection spans all stages.

### Stage 3 exception
Stage 3 uses separate param groups with different LRs, which `_run_stage` doesn't support. It has its own inline loop.

### Epoch budget
Total epochs = `n_epochs` (as before). For two_stage: `encoder_pretrain_epochs` + remainder. For three_stage: `encoder_pretrain_epochs` + flow_epochs + `joint_finetune_epochs`, where `flow_epochs = n_epochs - encoder_pretrain_epochs - joint_finetune_epochs`.

## Files modified

| File | Changes |
|------|---------|
| `scripts/kfold_age_inference.py` | `loss_mode` in forward, `linear` encoder, staged training, new CLI args |
| `kfold_age_inference.sh` | New params: `TRAINING_STAGES`, `ENCODER_PRETRAIN_EPOCHS`, `JOINT_FINETUNE_EPOCHS` |

## Experiments to run

1. **Linear encoder, joint:** `--encoder_type linear --training_stages joint` — does a learned linear projection beat PCA?
2. **MLP, two_stage:** `--encoder_type mlp --training_stages two_stage --encoder_pretrain_epochs 100 --n_epochs 200` — does decoupling fix the moving-target problem?
3. **MLP, three_stage:** `--encoder_type mlp --training_stages three_stage --encoder_pretrain_epochs 100 --n_epochs 250 --joint_finetune_epochs 50` — does fine-tuning after stabilization help further?
4. **Linear, two_stage:** `--encoder_type linear --training_stages two_stage` — linear encoder with stable flow training.
