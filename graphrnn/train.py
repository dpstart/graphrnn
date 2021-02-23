from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from time import gmtime, strftime
import time as tm
import numpy as np
import torch
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

from util import binary_cross_entropy_weight, get_graph, save_graph_list
from dataset import decode_adj, encode_adj


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, device="cpu"):
    """
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    """

    # do sigmoid first
    y = torch.sigmoid(y)
    # do sampling
    if sample:
        if sample_time > 1:
            y_result = torch.autograd.Variable(
                torch.rand(y.size(0), y.size(1), y.size(2))
            ).to(device)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = torch.autograd.Variable(
                        torch.rand(y.size(1), y.size(2))
                    ).to(device)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = torch.autograd.Variable(
                torch.rand(y.size(0), y.size(1), y.size(2))
            ).to(device)
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def train_mlp_epoch(
    epoch,
    args,
    rnn,
    output,
    data_loader,
    optimizer_rnn,
    optimizer_output,
    scheduler_rnn,
    scheduler_output,
    device,
):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data["x"].float()
        y_unsorted = data["y"].float()
        y_len_unsorted = data["len"]
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = torch.autograd.Variable(x).to(device)
        y = torch.autograd.Variable(y).to(device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = torch.nn.utils.rnn.pack_padded_sequence(
            y_pred, y_len, batch_first=True
        )
        y_pred = torch.nn.utils.rnn.pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if (
            epoch % args.epochs_log == 0 and batch_idx == 0
        ):  # only output first batch's statistics
            print(
                "Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}".format(
                    epoch,
                    args.epochs,
                    loss.item(),
                    args.graph_type,
                    args.num_layers,
                    args.hidden_size_rnn,
                )
            )

        loss_sum += loss.item()
    return loss_sum / (batch_idx + 1)


def test_mlp_epoch(
    epoch,
    args,
    rnn,
    output,
    test_batch_size=16,
    save_histogram=False,
    sample_time=1,
    device="cpu",
):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = torch.autograd.Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).to(
        device
    )  # normalized prediction score
    y_pred_long = torch.autograd.Variable(
        torch.zeros(test_batch_size, max_num_node, args.max_prev_node)
    ).to(
        device
    )  # discrete prediction
    x_step = torch.autograd.Variable(
        torch.ones(test_batch_size, 1, args.max_prev_node)
    ).to(device)
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i : i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(
            y_pred_step, sample=True, sample_time=sample_time, device=device
        )
        y_pred_long[:, i : i + 1, :] = x_step
        rnn.hidden = torch.autograd.Variable(rnn.hidden.data).to(device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def train(args, dataset_train, rnn, output, device):
    # check if load existing model

    epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(
        optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate
    )
    scheduler_output = MultiStepLR(
        optimizer_output, milestones=args.milestones, gamma=args.lr_rate
    )

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        train_mlp_epoch(
            epoch,
            args,
            rnn,
            output,
            dataset_train,
            optimizer_rnn,
            optimizer_output,
            scheduler_rnn,
            scheduler_output,
            device,
        )
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.test_total_size:
                    G_pred_step = test_mlp_epoch(
                        epoch,
                        args,
                        rnn,
                        output,
                        test_batch_size=args.test_batch_size,
                        sample_time=sample_time,
                        device=device,
                    )
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = (
                    args.graph_save_path
                    + args.fname_pred
                    + str(epoch)
                    + "_"
                    + str(sample_time)
                    + ".dat"
                )
                save_graph_list(G_pred, fname)
            print("test done, graphs saved")

        epoch += 1
