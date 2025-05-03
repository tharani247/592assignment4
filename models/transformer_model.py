import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # (1, max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer to output prediction
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Input x: shape (batch_size, seq_len, input_size)
        Output: shape (batch_size, 1)
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding (trim or extend to match sequence length)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Use the last time step’s output for prediction
        out = x[:, -1, :]  # shape (batch_size, d_model)
        out = self.decoder(out)  # shape (batch_size, 1)
        return out
