import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
    
class FluxDataset(Dataset):
    def __init__(self, flux_array, t_array, flux_err_array=None):
        self.flux = torch.tensor(flux_array, dtype=torch.float32)
        self.flux_err = None
        if flux_err_array is not None:
            self.flux_err = torch.tensor(flux_err_array, dtype=torch.float32)
        self.t = torch.tensor(t_array, dtype=torch.float32)

    def __len__(self):
        return self.flux.shape[0]

    def __getitem__(self, idx):
        x_flux = self.flux[idx].unsqueeze(-1)  # [seq_len, 1]
        x_t = self.t[idx].unsqueeze(-1)      # [seq_len, 1]
        if self.flux_err is not None:
            x_err = self.flux_err[idx].unsqueeze(-1)  # [seq_len, 1]
            x = torch.cat([x_flux, x_err], dim=-1)  # [seq_len, 2]
        else:
            x = x_flux

        # Input is both flux+error, but target is flux only
        return x, x_t, x_flux, x_err

class TimeChannelPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, min_period, max_period):
        super().__init__()
        self.embed_dim = embed_dim
        self.min_period = min_period
        self.max_period = max_period

        # Precompute the angular frequencies
        num_freqs = embed_dim // 2
        periods = torch.logspace(np.log10(min_period), np.log10(max_period), num_freqs)
        self.register_buffer("div_term", 2 * np.pi / periods)  # shape: (num_freqs,)

    def forward(self, x, t):
        """
        x: (batch, seq_len, embed_dim) - input embeddings
        t: (batch, seq_len, 1) - time values (real times)
        """
        batch_size, seq_len, _ = x.shape
        # Normalize time to [0,1] (optional but helps stability)
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)

        # Compute angles: (batch, seq_len, num_freqs)
        angles = t_norm * self.div_term.unsqueeze(0).unsqueeze(0)

        # Sin/cos: (batch, seq_len, embed_dim)
        pe = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)

        return x + pe

# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_dim, max_len,
#                  min_period, max_period):
#         super().__init__()
#         pe = torch.zeros(max_len, embed_dim)
#         position = torch.arange(0, max_len).unsqueeze(1)

#         # Frequencies sampled logarithmically between min and max
#         num_freqs = embed_dim // 2
#         periods = torch.logspace(
#             np.log10(min_period), np.log10(max_period), num_freqs
#         )
#         div_term = 2 * np.pi / periods  # angular frequencies

#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)

#         self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]



class TimeSeriesTransformer(nn.Module):
    def __init__(self, min_period, max_period, embed_dim=32, ff_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, embed_dim)
        self.pos_encoding = TimeChannelPositionalEncoding(embed_dim, min_period, max_period)
        self.encoder = TransformerEncoder(embed_dim, ff_dim, num_layers, dropout)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x, t):
        # x: (batch, seq_len, channels)  -> features
        # t: (batch, seq_len, 1)       -> time channel
        x = self.input_proj(x)

        # Positional encoding based on time channel
        x = self.pos_encoding(x, t)

        x, _ = self.encoder(x)
        x = self.output_proj(x)
        return x

    def positional_encoding(self, t, d_model):
            """
            t: (batch, seq_len, 1) - time values
            d_model: embedding dimension
            Returns: (batch, seq_len, d_model) positional encoding
            """
            # Normalize time to [0,1] to prevent large numbers
            t_min, t_max = t.min(), t.max()
            t_norm = (t - t_min) / (t_max - t_min + 1e-8)

            # Create log-spaced frequencies for multi-scale encoding
            half_dim = d_model // 2
            freqs = 1.0 / (10 ** torch.linspace(0, 1, half_dim, device=t.device))  # log-spaced

            # Outer product: (batch, seq_len, half_dim)
            angles = t_norm * freqs.unsqueeze(0).unsqueeze(0)

            # Sin and cos concatenation
            pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

            # If d_model is odd, pad one dimension
            if pe.size(-1) < d_model:
                pe = torch.cat([pe, torch.zeros(pe.size(0), pe.size(1), 1, device=t.device)], dim=-1)

            return pe

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        out = torch.bmm(attn_weights, V)
        return out, attn_weights


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = SingleHeadAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # --- Attention + Residual ---
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # --- Feedforward + Residual ---
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)  # collect attention maps if you want to inspect them
        return x, attn_maps


class MaskedTimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_dim=1,
                 d_model=128,
                 nhead=1,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        # Learnable mask token - one per feature dimension
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output projection back to original space
        self.output_projection = nn.Linear(d_model, 1)
    def forward(self, x, timestamps, mask=None):
        """
        Args:
            x: [batch, seq_len, input_dim] - input time series
            timestamps: [batch, seq_len] - actual timestamps for positional encoding
            mask: [batch, seq_len] - boolean mask, True for positions to mask
        Returns:
            output: [batch, seq_len, input_dim] - reconstructed time series
            mask: [batch, seq_len] - mask used (for loss calculation)
        """
        batch_size, seq_len, _ = x.shape
        # Project input to model dimension
        x_projected = self.input_projection(x)
        # Add positional encoding (assuming you have this implemented)
        pos_encoding = self.create_positional_encoding(timestamps, self.input_projection.out_features)
        x_with_pos = x_projected + pos_encoding
        # Apply mask tokens - replace masked positions with learnable mask token
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x_with_pos)
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # Keep positional encoding but replace values with mask token
            x_with_pos = torch.where(mask_expanded,
                                     mask_tokens + pos_encoding,  # mask token + positional info
                                     x_with_pos)
        # Pass through transformer
        transformer_out = self.transformer(x_with_pos)
        # Project back to input dimension
        output = self.output_projection(transformer_out)
        return output, mask
    def create_positional_encoding(self, timestamps, d_model):
        """
        Create sinusoidal positional encoding based on actual timestamps.
        Args:
            timestamps: [batch, seq_len] - actual time values
            d_model: dimension of model
        """
        batch_size, seq_len = timestamps.shape
        pe = torch.zeros(batch_size, seq_len, d_model, device=timestamps.device)
        # Create sinusoidal features
        div_term = torch.exp(torch.arange(0, d_model, 2, device=timestamps.device) *
                            -(np.log(10000.0) / d_model))
        # Expand timestamps for broadcasting
        pos = timestamps.unsqueeze(-1)  # [batch, seq_len, 1]
        # Apply sinusoidal transformation
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe
class MaskedReconstructionLoss(nn.Module):
    """Loss calculated only on masked positions"""
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: [batch, seq_len, features]
            targets: [batch, seq_len, features] - original unmasked values
            mask: [batch, seq_len] - boolean, True for masked positions
        """
        if mask is None or mask.sum() == 0:
            # No masking, return 0 loss (or could calculate on everything)
            return torch.tensor(0.0, device=predictions.device)
        # Extract only masked positions
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        if self.loss_type == 'mse':
            loss = F.mse_loss(masked_predictions, masked_targets)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(masked_predictions, masked_targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss
def create_random_block_mask(seq_len, mask_ratio=0.15, block_size=5):
    """
    Create random block mask for time series.
    Args:
        seq_len: sequence length
        mask_ratio: fraction of sequence to mask
        block_size: size of contiguous blocks to mask
    Returns:
        mask: [seq_len] boolean tensor, True for positions to mask
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    # Calculate number of blocks to mask
    n_masked = int(seq_len * mask_ratio)
    n_blocks = max(1, n_masked // block_size)
    # Randomly select block starting positions
    possible_starts = list(range(0, seq_len - block_size + 1))
    if len(possible_starts) > 0 and n_blocks > 0:
        block_starts = np.random.choice(
            possible_starts,
            size=min(n_blocks, len(possible_starts)),
            replace=False
        )
        # Mask blocks
        for start in block_starts:
            mask[start:start + block_size] = True
    return mask
# Training example
def train_step(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in data_loader:
        x, timestamps = batch  # x: [batch, seq_len, features]
        batch_size, seq_len, _ = x.shape
        # Create random masks for each sequence in batch
        masks = torch.stack([
            create_random_block_mask(seq_len, mask_ratio=0.3, block_size=5)
            for _ in range(batch_size)
        ])
        # Store original values for loss calculation
        x_original = x.clone()
        # Forward pass
        predictions, mask_used = model(x, timestamps, masks)
        # Calculate loss ONLY on masked positions
        loss = loss_fn(predictions, x_original, mask_used)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)