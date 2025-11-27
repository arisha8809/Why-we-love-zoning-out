# src/models/lstm.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2,
                 bidirectional=False, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers>1 else 0.0)
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, max(fc_in//2, 8)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(fc_in//2, 8), num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # use last timestep
        last = out[:, -1, :]  # (batch, hidden*dirs)
        logits = self.classifier(last)
        return logits
