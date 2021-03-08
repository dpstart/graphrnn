import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


import pytorch_lightning as pl

from util import binary_cross_entropy_weight, get_graph, save_graph_list, sample_sigmoid
from dataset import decode_adj, encode_adj

device = "cuda" if torch.cuda.is_available() else "cpu"



class GraphRNN(pl.LightningModule):

    def __init__(self, rnn, mlp, args):
        super().__init__()
        self.rnn = rnn
        self.mlp = mlp
        self.args = args
    
    def forward(self, x, y_len):
        h = self.rnn(x, pack=True, input_len=y_len)
        y_pred = self.mlp(h)
        return torch.sigmoid(y_pred)

    def test_epoch(
        self,
        epoch,
        test_batch_size=16,
        save_histogram=False,
        sample_time=1,

    ):
        self.rnn.hidden = self.rnn.init_hidden(test_batch_size)
        self.rnn.eval()
        self.mlp.eval()

        # generate graphs
        max_num_node = int(self.args.max_num_node)
        y_pred = torch.autograd.Variable(
            torch.zeros(test_batch_size, max_num_node, self.args.max_prev_node)
        ).to(device)  # normalized prediction score
        y_pred_long = torch.autograd.Variable(
            torch.zeros(test_batch_size, max_num_node, self.args.max_prev_node)
        ).to(device)  # discrete prediction
        x_step = torch.autograd.Variable(
            torch.ones(test_batch_size, 1, self.args.max_prev_node)
        ).to(device)
        for i in range(max_num_node):
            h = self.rnn(x_step)
            y_pred_step = self.mlp(h)
            y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid(
                y_pred_step, sample=True, sample_time=sample_time
            )
            y_pred_long[:, i : i + 1, :] = x_step
            self.rnn.hidden = torch.autograd.Variable(self.rnn.hidden.data)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        G_pred_list = []
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)

        return G_pred_list

    def training_step(self, batch, batch_idx):

        x_unsorted = batch["x"].float()
        y_unsorted = batch["y"].float()

        # Lens of original graphs
        y_len_unsorted = batch["len"]
        y_len_max = max(y_len_unsorted)

        # I think this is useless
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]

        # initialize lstm hidden state according to batch size
        self.rnn.hidden = self.rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.cpu().numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        y_pred = self(x, y_len)

        # UNCLEAR
        y_pred = torch.nn.utils.rnn.pack_padded_sequence(
            y_pred, y_len, batch_first=True
        )
        y_pred = torch.nn.utils.rnn.pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        return loss

    def configure_optimizers(self):
        
        optimizer = optim.Adam(list(self.rnn.parameters()) + list(self.mlp.parameters()), lr=self.args.lr)
        # scheduler = MultiStepLR(
        #     optimizer, milestones=self.args.milestones, gamma=self.args.lr_rate
        # )


        return optimizer

    def on_epoch_end(self):
         if self.current_epoch % self.args.epochs_test == 0 and self.current_epoch >= self.args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < self.args.test_total_size:
                    G_pred_step = self.test_epoch(
                        self.current_epoch,
                        test_batch_size=self.args.test_batch_size,
                        sample_time=sample_time
                    )
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = (
                    self.args.graph_save_path
                    + self.args.fname_pred
                    + str(self.current_epoch)
                    + "_"
                    + str(sample_time)
                    + ".dat"
                )
                save_graph_list(G_pred, self.args.graph_save_path, fname)
            print("test done, graphs saved")



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

        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.25)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("sigmoid"))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, h):
        return self.mlp(h)
