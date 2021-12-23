import torch
from torch import nn


class LSTM(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=False,
        dropout=0,
        device='cpu',
    ):
        super().__init__()

        self.cell = torch.nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional,
            dropout=dropout).to(
                device
            )  # Input to LSTM: conv_output + prev_action (+ prev_reward)

        # linear layer before cell
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU()).to(device)

    def forward(self, input, init_rnn_state=None, detach=False):
        input = self.mlp(input)

        self.cell.flatten_parameters()
        lstm_out, (hn, cn) = self.cell(input, init_rnn_state)
        if detach:
            lstm_out = lstm_out.detach()
        return lstm_out, (hn, cn)