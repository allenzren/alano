from model import *


class GRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        device='cpu',
        num_layers=1,
        bidirectional=False,
        dropout=0,
    ):
        super(GRU, self).__init__()
        self.cell = torch.nn.GRU(hidden_size,
                                 hidden_size,
                                 bias=True,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 dropout=dropout).to(device)

        # linear layer before cell
        self.mlp = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU()).to(device)

        # Set biases in reset gate to -1 to encourage memory
        for name, param in self.cell.named_parameters():
            if 'bias' in name:
                param.data[:hidden_size] = -1

    def forward(self, input, init_rnn_state=None, detach=False):
        input = self.mlp(input)

        if init_rnn_state is None:
            hn = None
        else:
            hn, _ = init_rnn_state  # dummy
        self.cell.flatten_parameters()
        gru_out, hn = self.cell(input, hn)
        if detach:
            gru_out = gru_out.detach()
        return gru_out, (hn, hn.clone().detach()
                         )  # dummy cell state to match lstm
