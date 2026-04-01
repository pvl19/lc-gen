"""
Metadata-based age predictor using MLP encoder and normalizing flow.

Architecture:
  1. MLP encoder: maps metadata features -> latent embedding
  2. Normalizing flow head: conditioned on [latent_embedding, Prot] to predict age

The model is trained with negative log-likelihood loss from the normalizing flow.
"""

import torch
import torch.nn as nn
import numpy as np

try:
    import zuko
except ImportError:
    zuko = None


# Default metadata fields to encode (excluding Prot which is a conditioning variable)
DEFAULT_METADATA_FIELDS = [
    'cadence_s',
    'Tmag',
    'sector',
    'camera',
    'ccd',
    'parallax',
    'parallax_error',
    'G0',
    'G0_err',
    'BPRP0',
    'BPRP0_err',
    'mean_flux',
    'std_flux',
]


class MetadataEncoder(nn.Module):
    """MLP encoder for metadata features with optional mask input.

    When use_mask=True, the encoder concatenates a binary mask to the input,
    allowing the model to distinguish between "value is 0" and "value is missing".
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.use_mask = use_mask
        self.input_dim = input_dim

        # If using mask, input dimension doubles (features + mask)
        actual_input_dim = input_dim * 2 if use_mask else input_dim

        layers = []
        prev_dim = actual_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode metadata to latent space.

        Args:
            x: (B, input_dim) metadata features
            mask: (B, input_dim) binary mask (1=valid, 0=missing), required if use_mask=True

        Returns:
            (B, latent_dim) latent embedding
        """
        if self.use_mask:
            if mask is None:
                # If no mask provided, assume all values are valid
                mask = torch.ones_like(x)
            # Concatenate features and mask
            x = torch.cat([x, mask], dim=1)

        return self.encoder(x)


class RandomFeatureMasker(nn.Module):
    """Randomly masks features during training for robustness to missing values.

    During training, samples the number of features to mask from a Poisson distribution,
    then randomly selects which features to mask. This teaches the model to make
    predictions even when some inputs are missing.

    During evaluation, no masking is applied.
    """

    def __init__(self, n_features: int, poisson_lambda: float = 2.0, mask_value: float = 0.0):
        """
        Args:
            n_features: number of input features
            poisson_lambda: mean of Poisson distribution for number of features to mask
            mask_value: value to use for masked features (typically 0)
        """
        super().__init__()
        self.n_features = n_features
        self.poisson_lambda = poisson_lambda
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Apply random masking during training.

        Args:
            x: (B, n_features) input features
            mask: (B, n_features) validity mask (1=valid, 0=missing)

        Returns:
            tuple of (masked_x, updated_mask)
        """
        if not self.training:
            # No masking during evaluation
            return x, mask

        B = x.shape[0]
        device = x.device

        # Sample number of features to mask for each sample
        # Clamp to [0, n_features-1] to ensure at least one feature remains
        n_to_mask = torch.poisson(torch.full((B,), self.poisson_lambda, device=device))
        n_to_mask = n_to_mask.clamp(0, self.n_features - 1).long()

        # Create additional mask for random masking
        random_mask = torch.ones_like(x)

        for i in range(B):
            n = n_to_mask[i].item()
            if n > 0:
                # Get indices of currently valid features
                valid_indices = torch.where(mask[i] > 0.5)[0]
                if len(valid_indices) > 1:  # Keep at least one feature
                    n = min(n, len(valid_indices) - 1)
                    # Randomly select features to mask
                    perm = torch.randperm(len(valid_indices), device=device)[:n]
                    mask_indices = valid_indices[perm]
                    random_mask[i, mask_indices] = 0.0

        # Apply random mask
        updated_mask = mask * random_mask
        masked_x = x * random_mask  # Zero out masked features

        return masked_x, updated_mask


class MetadataAgePredictor(nn.Module):
    """
    Predicts stellar age from metadata using a normalizing flow.

    The model consists of:
    1. MLP encoder: metadata features -> latent embedding (with optional mask input)
    2. Optional random feature masking during training for robustness
    3. NSF flow: conditioned on [latent_embedding, Prot] -> age distribution
    """

    def __init__(
        self,
        n_metadata_features: int = 11,
        latent_dim: int = 32,
        encoder_hidden_dims: list = None,
        encoder_dropout: float = 0.1,
        flow_transforms: int = 4,
        flow_hidden_features: list = None,
        use_mask: bool = False,
        use_random_masking: bool = False,
        mask_poisson_lambda: float = 2.0,
    ):
        """
        Args:
            n_metadata_features: number of input metadata features
            latent_dim: dimension of latent embedding
            encoder_hidden_dims: hidden layer sizes for encoder MLP
            encoder_dropout: dropout rate in encoder
            flow_transforms: number of flow transforms in NSF
            flow_hidden_features: hidden layer sizes in flow networks
            use_mask: if True, encoder accepts mask input to handle missing values
            use_random_masking: if True, randomly mask features during training
            mask_poisson_lambda: mean of Poisson distribution for random masking
        """
        super().__init__()

        if zuko is None:
            raise ImportError("zuko is required for normalizing flow. Install with: pip install zuko")

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [64, 64]
        if flow_hidden_features is None:
            flow_hidden_features = [64, 64]

        self.n_metadata_features = n_metadata_features
        self.latent_dim = latent_dim
        self.use_mask = use_mask
        self.use_random_masking = use_random_masking

        # MLP encoder for metadata
        self.encoder = MetadataEncoder(
            input_dim=n_metadata_features,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=encoder_dropout,
            use_mask=use_mask,
        )

        # Random feature masker for training augmentation
        self.random_masker = None
        if use_random_masking:
            self.random_masker = RandomFeatureMasker(
                n_features=n_metadata_features,
                poisson_lambda=mask_poisson_lambda,
            )

        # Flow context dimension: latent embedding + Prot (log-transformed)
        flow_context_dim = latent_dim + 1

        # Neural Spline Flow for age prediction
        # Predicts log10(age in Myr) since age spans orders of magnitude
        # log10 ages range from ~0.2 (1.6 Myr) to ~3.6 (4000 Myr)
        self.flow = zuko.flows.NSF(
            features=1,  # 1D output (log10 age)
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
        # Ensure prot is 2D
        if prot.dim() == 1:
            prot = prot.unsqueeze(1)

        # Initialize mask if not provided
        if mask is None:
            mask = torch.ones_like(metadata)

        # Apply random masking during training
        if self.random_masker is not None and self.training:
            metadata, mask = self.random_masker(metadata, mask)

        # Encode metadata (with mask if use_mask=True)
        if self.use_mask:
            latent = self.encoder(metadata, mask)
        else:
            latent = self.encoder(metadata)

        # Build flow context: [latent, log10(Prot)]
        context = torch.cat([latent, prot], dim=1)

        # Get flow distribution conditioned on context
        dist = self.flow(context)

        result = {'latent': latent}

        if age is not None:
            # Ensure age is 2D for flow
            if age.dim() == 1:
                age = age.unsqueeze(1)

            # Compute log probability
            log_prob = dist.log_prob(age)
            result['log_prob'] = log_prob.view(-1)  # Ensure 1D
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

        # Encode with mask if using mask input
        if self.use_mask:
            if mask is None:
                mask = torch.ones_like(metadata)
            latent = self.encoder(metadata, mask)
        else:
            latent = self.encoder(metadata)

        context = torch.cat([latent, prot], dim=1)
        dist = self.flow(context)

        # Sample from flow
        samples = dist.sample((n_samples,))  # (n_samples, B, 1)
        samples = samples.squeeze(-1).T  # (B, n_samples)

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
        Predict age with uncertainty estimation via sampling.

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
        samples = self.sample(metadata, prot, n_samples, mask=mask)  # (B, n_samples)

        result = {
            'log_age_mean': samples.mean(dim=1),
            'log_age_samples': samples,
        }

        if return_std:
            result['log_age_std'] = samples.std(dim=1)

        # Convert to linear age space
        age_samples = 10 ** samples  # Convert from log10 to linear
        result['age_mean'] = age_samples.mean(dim=1)
        result['age_median'] = age_samples.median(dim=1).values
        result['age_std'] = age_samples.std(dim=1)

        return result


class MetadataStandardizer:
    """
    Standardizes metadata features for model input using fixed normalization rules.

    This standardizer uses deterministic transformations that don't depend on
    training data distribution, making it portable across different datasets.

    Normalization rules:
        - sector: divide by 100
        - cadence_s, Tmag, parallax, parallax_error,
          G0, G0_err, BPRP0_err, mean_flux, std_flux: log10 transform
        - BPRP0, camera, ccd: raw (no transformation)
    """

    # Fields that get log10 transformed
    LOG10_FIELDS = {
        'cadence_s',
        'Tmag',
        'parallax',
        'parallax_error',
        'G0',
        'G0_err',
        'BPRP0_err',
        'mean_flux',
        'std_flux',
    }

    # Fields that get divided by a constant
    SCALE_FIELDS = {
        'sector': 100.0,
    }

    # Fields that are passed through raw
    RAW_FIELDS = {
        'BPRP0',
        'camera',
        'ccd',
    }

    def __init__(self, fields: list = None):
        self.fields = fields if fields is not None else DEFAULT_METADATA_FIELDS

    def transform(self, data: dict, return_mask: bool = False):
        """Transform metadata to normalized features.

        Args:
            data: dict mapping field names to arrays of values
            return_mask: if True, also return validity mask

        Returns:
            If return_mask=False: (N, n_fields) array of normalized features
            If return_mask=True: tuple of (features, mask) where mask is 1 for valid, 0 for missing
        """
        n_samples = len(data[self.fields[0]])
        features = np.zeros((n_samples, len(self.fields)), dtype=np.float32)
        mask = np.ones((n_samples, len(self.fields)), dtype=np.float32)

        for i, field in enumerate(self.fields):
            values = np.array(data[field], dtype=np.float32)

            if field in self.LOG10_FIELDS:
                # Log10 transform (handle non-positive values)
                values = np.where(values > 0, np.log10(values), np.nan)
            elif field in self.SCALE_FIELDS:
                # Divide by scale factor
                values = values / self.SCALE_FIELDS[field]
            # else: RAW_FIELDS - no transformation

            # Track which values are missing (NaN)
            is_missing = np.isnan(values)
            mask[:, i] = (~is_missing).astype(np.float32)

            # Fill NaN with 0
            features[:, i] = np.nan_to_num(values, nan=0.0)

        if return_mask:
            return features, mask
        return features

    def inverse_transform_field(self, field: str, values: np.ndarray) -> np.ndarray:
        """Inverse transform a single field back to original scale.

        Args:
            field: field name
            values: normalized values

        Returns:
            values in original scale
        """
        if field in self.LOG10_FIELDS:
            return 10 ** values
        elif field in self.SCALE_FIELDS:
            return values * self.SCALE_FIELDS[field]
        else:
            return values

    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            'fields': self.fields,
        }

    def load_state_dict(self, state: dict):
        """Load state from saved dict."""
        self.fields = state['fields']
        return self

    def get_field_info(self) -> dict:
        """Get information about how each field is transformed."""
        info = {}
        for field in self.fields:
            if field in self.LOG10_FIELDS:
                info[field] = 'log10'
            elif field in self.SCALE_FIELDS:
                info[field] = f'divide by {self.SCALE_FIELDS[field]}'
            else:
                info[field] = 'raw'
        return info
