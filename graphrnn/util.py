import os

import torch
import torch.nn.functional as F

import numpy as np
import pickle
import networkx as nx

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
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
            )
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = torch.autograd.Variable(
                        torch.rand(y.size(1), y.size(2))
                    )
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data > 0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = torch.autograd.Variable(
                torch.rand(y.size(0), y.size(1), y.size(2))
            )
            y_result = torch.gt(y, y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2)) * thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def binary_cross_entropy_weight(
    y_pred, y, has_weight=False, weight_length=1, weight_max=10
):
    """
    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    """
    if has_weight:
        weight = torch.ones(y.size(0), y.size(1), y.size(2))
        weight_linear = torch.arange(1, weight_length + 1) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(
            y.size(0), 1, y.size(2)
        )
        weight[:, -1 * weight_length :, :] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.cuda())
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def get_graph(adj):
    """
    get a graph from zero-padded adj
    :param adj:
    :return:
    """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


# save a list of graphs
def save_graph_list(G_list, path, fname):

    if not os.path.exists(path):
        os.makedirs(path)

    with open(fname, "wb") as f:
        pickle.dump(G_list, f)
