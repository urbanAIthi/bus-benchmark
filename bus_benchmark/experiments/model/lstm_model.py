import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        n_output_timesteps: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_output_timesteps = n_output_timesteps
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.decoder = nn.LSTM(
            input_size=n_output_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.out_proj = nn.Linear(hidden_size, n_output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        _, (h_enc, c_enc) = self.encoder(x)
        h_dec = h_enc.view(self.num_layers, 2, batch_size, self.hidden_size).sum(dim=1)
        c_dec = c_enc.view(self.num_layers, 2, batch_size, self.hidden_size).sum(dim=1)

        dec_input = torch.zeros(
            batch_size, 1, self.out_proj.out_features, device=x.device
        )

        outputs = []
        for _ in range(self.n_output_timesteps):
            dec_out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            pred = self.out_proj(dec_out.squeeze(1))
            outputs.append(pred.unsqueeze(1))
            dec_input = pred.unsqueeze(1)

        return torch.cat(outputs, dim=1)
