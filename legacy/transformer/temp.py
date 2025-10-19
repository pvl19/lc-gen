class TimeSeriesTransformer(nn.Module):
    def __init__(self, embed_dim=32, ff_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, ff_dim, num_layers, dropout)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x, t):
        # x: (batch, seq_len, channels)  -> features
        # t: (batch, seq_len, 1)       -> time channel
        x = self.input_proj(x)

        # Positional encoding based on time channel
        x = self.positional_encoding(x, t)

        x, _ = self.encoder(x)
        x = self.output_proj(x)
        return x

    def positional_encoding(self, x, t):
        """
        x: (batch, seq_len, embed_dim)
        t: (batch, seq_len, 1)   -> time values, e.g., in days or seconds
        """
        # normalize time to 0..1 (optional but helps stability)
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)

        batch_size, seq_len, d_model = x.shape
        freqs = 1.0 / (10 ** torch.linspace(0, 1, d_model // 2))  # log-spaced

        # outer product: (batch, seq_len, d_model/2)
        angles = t_norm.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)

        # sin/cos
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return x + pe