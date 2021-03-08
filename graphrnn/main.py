import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt

import argparse
import networkx as nx
import random

from graph import create_graph
from model import GRU, MLP, GraphRNN
from dataset import Graph_sequence_sampler_pytorch


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("graph_type", nargs="?", default="grid_small")
parser.add_argument("embedding_size_rnn", nargs="?", default=64)
parser.add_argument("hidden_size_rnn", nargs="?", default=128)
parser.add_argument("embedding_size_output", nargs="?", default=64)
parser.add_argument("num_layers", nargs="?", default=4)
parser.add_argument("batch_size", nargs="?", default=32)
parser.add_argument("batch_ratio", nargs="?", default=32)
parser.add_argument("num_workers", nargs="?", default=4)
parser.add_argument("max_num_node", nargs="?", default=None)
parser.add_argument("lr", nargs="?", default=0.003)
parser.add_argument("milestones", nargs="?", default=[400, 1000])
parser.add_argument("lr_rate", nargs="?", default=0.3)
parser.add_argument("epochs", nargs="?", default=3000)
parser.add_argument("epochs_log", nargs="?", default=1)
parser.add_argument("epochs_test", nargs="?", default=100)
parser.add_argument("epochs_test_start", nargs="?", default=1)
parser.add_argument("milestones", nargs="?", default=[400, 1000])
parser.add_argument("test_total_size", nargs="?", default=10)
parser.add_argument("test_batch_size", nargs="?", default=32)
parser.add_argument("graph_save_path", nargs="?", default="./graphs/")
parser.add_argument("fname_pred", nargs="?", default="out_")

args = parser.parse_args()

graphs = create_graph(args)
args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])

### Split dataset

random.seed(42)
random.shuffle(graphs)
graphs_len = len(graphs)
graphs_test = graphs[int(0.8 * graphs_len) :]
graphs_train = graphs[0 : int(0.8 * graphs_len)]
graphs_val = graphs[0 : int(0.2 * graphs_len)]

### Build models

rnn = GRU(
    input_size=args.max_prev_node,
    embedding_size=args.embedding_size_rnn,
    hidden_size=args.hidden_size_rnn,
    num_layers=args.num_layers,
    has_input=True,
    has_output=False,
).to(device)

output = MLP(
    h_size=args.hidden_size_rnn,
    embedding_size=args.embedding_size_output,
    y_size=args.max_prev_node,
).to(device)

### Build data stuff

dataset = Graph_sequence_sampler_pytorch(
    graphs_train, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node
)
sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    [1.0 / len(dataset) for i in range(len(dataset))],
    num_samples=args.batch_size * args.batch_ratio,
    replacement=True,
)
dataset_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    sampler=sample_strategy,
)

model = GraphRNN(rnn, output, args)

### Train
trainer = pl.Trainer(gpus=1)
trainer.fit(model, dataset_loader)
