"""Train-time metadata masking, adapted from DOROTHY's dynamic hierarchical
masking (see docs/from_josh.md).

Two independent masking levels are applied per batch, train-only:

  1. Block level — per star, an independent Bernoulli(p_block) decides whether
     the whole metadata embedding is suppressed. (NOT DOROTHY's survey-level
     code: with a single metadata group, DOROTHY's guaranteed-keeper argmax
     would force-keep it every time, so block-drop must be a plain Bernoulli.)

  2. Field level — DOROTHY-style hierarchical masking over all metadata fields:
     a per-batch keep probability p_keep ~ U(p_keep_min, p_keep_max); each
     (star, field) is kept by an independent Bernoulli(p_keep); one
     guaranteed-keeper field per star is force-kept (DOROTHY trick A: uniform
     random choice via argmax over the random tensor) so a non-block-dropped
     star always retains >= 1 field.

Block-dropped stars skip the field step — they are already fully suppressed.

Masked field        -> value set to 0, mask channel set to 0.
Block-dropped star  -> all fields value 0, mask 0, and block_drop flag set so
                       the model zeroes that star's metadata embedding.

All masking is vectorised over the batch. Mixed RNG by design: a single
per-batch CPU scalar via numpy for p_keep, per-(star,field) draws via torch on
the data's device.
"""
import numpy as np
import torch


class DynamicMetadataMasking:
    """See module docstring. Instantiate once; call per batch during training."""

    def __init__(self, p_block: float = 0.15,
                 p_keep_min: float = 0.3, p_keep_max: float = 1.0):
        if not 0.0 <= p_block <= 1.0:
            raise ValueError(f'p_block must be in [0, 1], got {p_block}')
        if not 0.0 <= p_keep_min <= p_keep_max <= 1.0:
            raise ValueError(
                f'require 0 <= p_keep_min <= p_keep_max <= 1, '
                f'got ({p_keep_min}, {p_keep_max})')
        self.p_block = p_block
        self.p_keep_min = p_keep_min
        self.p_keep_max = p_keep_max

    def __call__(self, metadata: torch.Tensor, meta_mask: torch.Tensor = None):
        """Apply masking to one batch of metadata.

        Args:
            metadata:  (B, M) standardized metadata tensor.
            meta_mask: (B, M) validity mask (1 = present). None -> all ones
                       (the dataset currently provides no missingness mask).

        Returns:
            metadata_out  (B, M)  — masked values (masked field -> 0)
            meta_mask_out (B, M)  — 1 = field present, 0 = masked
            block_drop    (B,)    — bool, True where the whole encoder is dropped
        """
        if metadata.dim() != 2:
            raise ValueError(f'expected (B, M) metadata, got {tuple(metadata.shape)}')
        B, M = metadata.shape
        device = metadata.device
        if meta_mask is None:
            meta_mask = torch.ones_like(metadata)

        # --- Block level: independent Bernoulli(p_block) per star ---
        block_drop = torch.rand(B, device=device) < self.p_block          # (B,)

        # --- Field level: DOROTHY-style, single per-batch keep probability ---
        p_keep = float(np.random.uniform(self.p_keep_min, self.p_keep_max))
        rand = torch.rand(B, M, device=device)
        keep = rand < p_keep                                              # (B, M)
        # Guaranteed keeper: force-keep the field with the largest random value
        # per star (vectorised uniform random choice — DOROTHY trick A).
        guaranteed = rand.argmax(dim=1)                                   # (B,)
        keep[torch.arange(B, device=device), guaranteed] = True
        # Block-dropped stars: every field masked (overrides the field level).
        keep = keep & (~block_drop).unsqueeze(1)                          # (B, M)

        keep_f = keep.to(metadata.dtype)
        metadata_out = metadata * keep_f
        meta_mask_out = meta_mask * keep_f
        return metadata_out, meta_mask_out, block_drop
