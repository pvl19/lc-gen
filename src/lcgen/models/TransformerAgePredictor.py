"""
Transformer-based age predictor using self-attention for metadata encoding.

This module provides an alternative to the MLP-based MetadataAgePredictor,
using a Transformer encoder instead. The normalizing flow prediction head
remains the same.

================================================================================
TRANSFORMER ARCHITECTURE EXPLANATION
================================================================================

A Transformer is a neural network architecture that uses "attention" to learn
relationships between different parts of the input. Originally designed for
sequences (like sentences), we adapt it here for tabular metadata.

KEY CONCEPTS:

1. SELF-ATTENTION
   ---------------
   Self-attention allows each feature to "look at" all other features and
   decide how much to weight each one. For example, when encoding Tmag
   (TESS magnitude), the model might learn to pay attention to parallax
   (which relates to distance) to better understand absolute brightness.

   Mathematically, for each feature we compute:
   - Query (Q): "What am I looking for?"
   - Key (K): "What do I contain?"
   - Value (V): "What information do I provide?"

   Attention weights = softmax(Q · K^T / sqrt(d))
   Output = Attention weights · V

2. MULTI-HEAD ATTENTION
   ---------------------
   Instead of one attention mechanism, we use multiple "heads" in parallel.
   Each head can learn different types of relationships:
   - Head 1 might learn: magnitude <-> parallax relationships
   - Head 2 might learn: color <-> temperature relationships
   - Head 3 might learn: flux <-> error relationships

   The outputs are concatenated and projected back.

3. FEED-FORWARD NETWORK (FFN)
   --------------------------
   After attention, each feature goes through a small MLP:
   FFN(x) = Linear(GELU(Linear(x)))

   This adds non-linearity and allows feature-specific transformations.

4. LAYER NORMALIZATION & RESIDUAL CONNECTIONS
   ------------------------------------------
   Each sub-layer (attention, FFN) has:
   - Residual connection: output = sublayer(x) + x
   - Layer normalization: stabilizes training

   This helps gradients flow and enables deeper networks.

5. POSITIONAL ENCODING
   --------------------
   Unlike sequences, our metadata features have no inherent order.
   We use learned positional embeddings so the model can distinguish
   "feature 0" from "feature 5", allowing it to learn feature-specific
   patterns.

6. CLS TOKEN (Classification Token)
   ---------------------------------
   We prepend a special learnable [CLS] token to the sequence.
   After passing through transformer layers, this token aggregates
   information from all features and serves as the final representation.
   This is a common technique from BERT-style models.

ARCHITECTURE DIAGRAM:

    Input: 11 metadata features (each a scalar)
           ↓
    ┌──────────────────────────────────────────┐
    │  Feature Embedding Layer                  │
    │  Each feature → d_model dimensional vector│
    │  (11 features → 11 × d_model)            │
    └──────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────┐
    │  Add [CLS] Token                          │
    │  Prepend learnable token                  │
    │  (11 → 12 tokens)                        │
    └──────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────┐
    │  Add Positional Embeddings                │
    │  Learned position for each of 12 slots   │
    └──────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────┐
    │  Transformer Encoder (× n_layers)        │
    │  ┌────────────────────────────────────┐  │
    │  │ Multi-Head Self-Attention          │  │
    │  │ (features attend to each other)    │  │
    │  └────────────────────────────────────┘  │
    │           ↓                              │
    │  ┌────────────────────────────────────┐  │
    │  │ Feed-Forward Network               │  │
    │  │ (per-feature transformation)       │  │
    │  └────────────────────────────────────┘  │
    └──────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────┐
    │  Extract [CLS] Token Output               │
    │  → latent embedding (d_model dims)       │
    └──────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────┐
    │  Project to latent_dim                    │
    │  d_model → latent_dim                    │
    └──────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────┐
    │  Normalizing Flow Head                    │
    │  Conditioned on [latent, log10(Prot)]    │
    │  → p(log10(age) | context)               │
    └──────────────────────────────────────────┘

COMPARISON TO MLP:

    MLP:         [f1, f2, ..., f11] → Linear → Linear → latent
                 (treats all features as a flat vector)

    Transformer: [f1, f2, ..., f11] → Attention → FFN → [CLS] → latent
                 (learns pairwise feature interactions explicitly)

The Transformer can learn more complex feature interactions but has more
parameters and may require more data to train effectively.

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import math

try:
    import zuko
except ImportError:
    zuko = None

from .MetadataAgePredictor import DEFAULT_METADATA_FIELDS, MetadataStandardizer, RandomFeatureMasker


class MetadataTransformerEncoder(nn.Module):
    """
    Transformer encoder for metadata features with attention masking for missing values.

    Treats each metadata field as a "token" and uses self-attention
    to learn relationships between features. Missing features are masked
    out in the attention computation (their attention weights become 0).
    """

    def __init__(
        self,
        n_features: int = 11,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        latent_dim: int = 32,
    ):
        """
        Args:
            n_features: number of input metadata features (11 by default)
            d_model: dimension of token embeddings (must be divisible by n_heads)
            n_heads: number of attention heads
            n_layers: number of transformer encoder layers
            d_ff: dimension of feed-forward hidden layer
            dropout: dropout rate
            latent_dim: output dimension of the encoder
        """
        super().__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.n_features = n_features
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Feature embedding: project each scalar feature to d_model dimensions
        # We use a separate linear layer for each feature to allow feature-specific
        # transformations (similar to having different "vocabularies" per feature)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

        # Learnable [CLS] token - will aggregate information from all features
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embeddings for n_features + 1 (for CLS token)
        # These help the model know which feature is which
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features + 1, d_model) * 0.02)

        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input shape: (batch, seq, features)
            norm_first=True,   # Pre-norm architecture (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Project to latent dimension
        self.output_proj = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode metadata features using transformer with optional attention masking.

        Args:
            x: (B, n_features) tensor of metadata values
            mask: (B, n_features) validity mask (1=valid, 0=missing), optional
                  Missing features will be excluded from attention computation.

        Returns:
            (B, latent_dim) latent embedding
        """
        B = x.shape[0]
        device = x.device

        # Embed each feature separately: (B, n_features) -> (B, n_features, d_model)
        embedded = []
        for i in range(self.n_features):
            # Extract feature i and add feature dimension: (B,) -> (B, 1)
            feat = x[:, i:i+1]
            # Embed: (B, 1) -> (B, d_model)
            emb = self.feature_embeddings[i](feat)
            embedded.append(emb)

        # Stack: list of (B, d_model) -> (B, n_features, d_model)
        embedded = torch.stack(embedded, dim=1)

        # Prepend [CLS] token: (B, n_features, d_model) -> (B, n_features+1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, d_model) -> (B, 1, d_model)
        embedded = torch.cat([cls_tokens, embedded], dim=1)

        # Add positional embeddings
        embedded = embedded + self.pos_embedding

        # Apply dropout
        embedded = self.embed_dropout(embedded)

        # Create attention mask for missing features
        # PyTorch TransformerEncoder expects src_key_padding_mask: (B, seq_len)
        # where True = masked (excluded from attention), False = valid
        src_key_padding_mask = None
        if mask is not None:
            # Prepend 1 for [CLS] token (always valid)
            cls_mask = torch.ones(B, 1, device=device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, n_features+1)
            # Convert to boolean: True = masked out (invalid)
            src_key_padding_mask = (full_mask < 0.5)

        # Pass through transformer encoder with attention mask
        # Shape: (B, n_features+1, d_model) -> (B, n_features+1, d_model)
        transformed = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)

        # Extract [CLS] token output (first position)
        cls_output = transformed[:, 0, :]  # (B, d_model)

        # Final normalization and projection
        cls_output = self.final_norm(cls_output)
        latent = self.output_proj(cls_output)  # (B, latent_dim)

        return latent


class TransformerAgePredictor(nn.Module):
    """
    Predicts stellar age from metadata using a Transformer encoder and normalizing flow.

    Architecture:
    1. Transformer encoder: metadata features -> latent embedding (with attention masking)
    2. Optional random feature masking during training for robustness
    3. NSF flow: conditioned on [latent_embedding, Prot] -> age distribution

    This is an alternative to MetadataAgePredictor that uses self-attention
    instead of an MLP to encode metadata features. The transformer naturally
    handles missing values through attention masking.
    """

    def __init__(
        self,
        n_metadata_features: int = 11,
        latent_dim: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        encoder_dropout: float = 0.1,
        flow_transforms: int = 4,
        flow_hidden_features: list = None,
        use_random_masking: bool = False,
        mask_poisson_lambda: float = 2.0,
    ):
        """
        Args:
            n_metadata_features: number of input metadata features
            latent_dim: dimension of latent embedding (output of encoder)
            d_model: dimension of transformer token embeddings
            n_heads: number of attention heads
            n_layers: number of transformer encoder layers
            d_ff: dimension of feed-forward hidden layer in transformer
            encoder_dropout: dropout rate in encoder
            flow_transforms: number of flow transforms in NSF
            flow_hidden_features: hidden layer sizes in flow networks
            use_random_masking: if True, randomly mask features during training
            mask_poisson_lambda: mean of Poisson distribution for random masking
        """
        super().__init__()

        if zuko is None:
            raise ImportError("zuko is required for normalizing flow. Install with: pip install zuko")

        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        self.n_metadata_features = n_metadata_features
        self.latent_dim = latent_dim
        self.use_random_masking = use_random_masking

        # Transformer encoder for metadata
        self.encoder = MetadataTransformerEncoder(
            n_features=n_metadata_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=encoder_dropout,
            latent_dim=latent_dim,
        )

        # Random feature masker for training augmentation
        self.random_masker = None
        if use_random_masking:
            self.random_masker = RandomFeatureMasker(
                n_features=n_metadata_features,
                poisson_lambda=mask_poisson_lambda,
            )

        # Flow context dimension: latent embedding + Prot (log10-transformed)
        flow_context_dim = latent_dim + 1

        # Neural Spline Flow for age prediction
        # Predicts log10(age in Myr)
        self.flow = zuko.flows.NSF(
            features=1,
            context=flow_context_dim,
            transforms=flow_transforms,
            hidden_features=flow_hidden_features,
        )

    def forward(
        self,
        metadata: torch.Tensor,
        prot: torch.Tensor,
        age: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            metadata: (B, n_metadata_features) metadata features (normalized)
            prot: (B,) or (B, 1) rotation period (log10-transformed)
            age: (B,) or (B, 1) target age in Myr (log10-transformed), optional
            mask: (B, n_metadata_features) validity mask (1=valid, 0=missing), optional

        Returns:
            dict with:
                'latent': (B, latent_dim) latent embedding
                'nll': scalar negative log-likelihood (if age provided)
                'log_prob': (B,) log probability of age (if age provided)
        """
        if prot.dim() == 1:
            prot = prot.unsqueeze(1)

        # Initialize mask if not provided
        if mask is None:
            mask = torch.ones_like(metadata)

        # Apply random masking during training
        if self.random_masker is not None and self.training:
            metadata, mask = self.random_masker(metadata, mask)

        # Encode metadata using transformer with attention masking
        latent = self.encoder(metadata, mask)

        # Build flow context: [latent, log10(Prot)]
        context = torch.cat([latent, prot], dim=1)

        # Get flow distribution conditioned on context
        dist = self.flow(context)

        result = {'latent': latent}

        if age is not None:
            if age.dim() == 1:
                age = age.unsqueeze(1)

            log_prob = dist.log_prob(age)
            result['log_prob'] = log_prob.view(-1)
            result['nll'] = -log_prob.mean()

        return result

    def sample(
        self,
        metadata: torch.Tensor,
        prot: torch.Tensor,
        n_samples: int = 1,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sample ages from the flow.

        Args:
            metadata: (B, n_metadata_features) metadata features
            prot: (B,) or (B, 1) rotation period (log10-transformed)
            n_samples: number of samples per input
            mask: (B, n_metadata_features) validity mask, optional

        Returns:
            (B, n_samples) sampled log10(age) values
        """
        if prot.dim() == 1:
            prot = prot.unsqueeze(1)

        # Encode with attention mask for missing features
        latent = self.encoder(metadata, mask)
        context = torch.cat([latent, prot], dim=1)
        dist = self.flow(context)

        samples = dist.sample((n_samples,))
        samples = samples.squeeze(-1).T

        return samples

    def predict(
        self,
        metadata: torch.Tensor,
        prot: torch.Tensor,
        n_samples: int = 100,
        return_std: bool = True,
        mask: torch.Tensor = None,
    ) -> dict:
        """
        Predict age with uncertainty estimation via Monte Carlo sampling.

        Args:
            metadata: (B, n_metadata_features) metadata features
            prot: (B,) or (B, 1) rotation period (log10-transformed)
            n_samples: number of Monte Carlo samples
            return_std: whether to return standard deviation
            mask: (B, n_metadata_features) validity mask, optional

        Returns:
            dict with:
                'log_age_mean': (B,) mean log10(age)
                'log_age_std': (B,) std of log10(age) (if return_std)
                'age_mean': (B,) mean age in Myr
                'age_median': (B,) median age in Myr
        """
        samples = self.sample(metadata, prot, n_samples, mask=mask)

        result = {
            'log_age_mean': samples.mean(dim=1),
            'log_age_samples': samples,
        }

        if return_std:
            result['log_age_std'] = samples.std(dim=1)

        # Convert from log10 to linear age
        age_samples = 10 ** samples
        result['age_mean'] = age_samples.mean(dim=1)
        result['age_median'] = age_samples.median(dim=1).values
        result['age_std'] = age_samples.std(dim=1)

        return result


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_architectures():
    """Compare MLP vs Transformer architectures."""
    from .MetadataAgePredictor import MetadataAgePredictor

    # Create both models with similar latent dimensions
    mlp_model = MetadataAgePredictor(
        n_metadata_features=11,
        latent_dim=32,
        encoder_hidden_dims=[64, 64],
    )

    transformer_model = TransformerAgePredictor(
        n_metadata_features=11,
        latent_dim=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )

    print("Architecture Comparison:")
    print(f"  MLP parameters:         {count_parameters(mlp_model):,}")
    print(f"  Transformer parameters: {count_parameters(transformer_model):,}")

    # Test forward pass
    x = torch.randn(4, 11)  # 4 samples, 11 features
    prot = torch.randn(4)
    age = torch.randn(4)

    mlp_out = mlp_model(x, prot, age)
    transformer_out = transformer_model(x, prot, age)

    print(f"\n  MLP latent shape:         {mlp_out['latent'].shape}")
    print(f"  Transformer latent shape: {transformer_out['latent'].shape}")
    print(f"\n  MLP NLL:         {mlp_out['nll'].item():.4f}")
    print(f"  Transformer NLL: {transformer_out['nll'].item():.4f}")


if __name__ == "__main__":
    compare_architectures()
