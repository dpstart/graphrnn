import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available else "cpu"


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        num_layers,
        has_input=True,
        has_output=False,
        output_size=None,
    ):

        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_input = has_input
        self.has_output = has_output

        self.hidden = None

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.gru = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
            )

        if has_output:
            self.output = nn.Sequential(
                *[
                    nn.Linear(input_size, embedding_size),
                    nn.ReLU(),
                    nn.Linear(embedding_size, output_size),
                ]
            )

    def init_hidden(self, batch_size):
        return torch.autograd.Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        ).to(device)

    def forward(self, x, pack=False, input_len=None):

        if self.has_input:

            x = self.input(x)
            x = F.relu(x)

        if pack:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, input_len, batch_first=True)
        out, self.hidden = self.gru(x, self.hidden)

        if pack:
            out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        if self.has_output:
            out = self.output(out)
        return out


class MLP(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):

        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Linear(h_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, y_size),
            ]
        )

    def forward(self, h):
        return self.mlp(h)
