#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import glob

import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

names = ['A',
         'graph_indicator', 'graph_labels', 'graph_attributes',
         'node_labels', 'node_attributes',
         'edge_labels', 'edge_attributes',
         'node_importance_labels'
         ]

def my_read_tu_data(folder, prefix):

    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1

    # graph_indicator subgraph id begin from 0
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1
    node_attributes = node_labels = important_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')


    if 'node_importance_labels' in names:
        important_labels = read_file(folder, prefix, 'node_importance_labels', torch.long)
        if important_labels.dim() == 1:
            important_labels = important_labels.unsqueeze(-1)
        important_labels = important_labels - important_labels.min(dim=0)[0]
        important_labels = important_labels.unbind(dim=-1)
        important_labels = [F.one_hot(x, num_classes=-1) for x in important_labels]
        important_labels = torch.cat(important_labels, dim=-1).to(torch.float)

    x = cat([node_attributes, important_labels])

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')

    edge_attr = edge_attributes.to(torch.float32)

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, )
    data, slices = split(data, batch)

    return data, slices


def msy_read_tu_data(folder, prefix, x_, edge_index_):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = important_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')

    if 'node_importance_labels' in names:
        important_labels = read_file(folder, prefix, 'node_importance_labels', torch.long)
        if important_labels.dim() == 1:
            important_labels = important_labels.unsqueeze(-1)
        important_labels = important_labels - important_labels.min(dim=0)[0]
        important_labels = important_labels.unbind(dim=-1)
        important_labels = [F.one_hot(x, num_classes=-1) for x in important_labels]
        important_labels = torch.cat(important_labels, dim=-1).to(torch.float)

    x = cat([node_attributes, important_labels])

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    edge_attr = edge_attributes.to(torch.float32)

    y = None
    if 'graph_attributes' in names:  # Regression problem.
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:  # Classification problem.
        y = read_file(folder, prefix, 'graph_labels', torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    data = Data(x=x_, edge_index=edge_index_, edge_attr=edge_attr, y=y)
    data, slices = split(data, batch)
    return data, slices

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    batch_row_tensor = torch.from_numpy(np.bincount(batch[row])).to(device=data.edge_index.device)

    edge_slice = torch.cumsum(batch_row_tensor, 0)
    edge_slice = torch.cat([torch.tensor([0], device=edge_slice.device), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices



if __name__ == '__main__':
    folder = '../data/eth/ico-wallets-old-new/2hop-20/Volume/raw'
    prefix = 'ETHG'

    data, slices = my_read_tu_data(folder, prefix)
    print(data)
    print(slices)
