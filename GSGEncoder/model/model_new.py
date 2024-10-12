#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F


class Project_Head(torch.nn.Module):
    # Todo: add num_layers

    def __init__(self, in_channels):
        super(Project_Head, self).__init__()

        self.block = Sequential(Linear(in_channels, in_channels),
                                BatchNorm1d(in_channels), ReLU(inplace=False),
                                Linear(in_channels, in_channels),
                                BatchNorm1d(in_channels), ReLU(inplace=False),
                                )

        self.linear_shortcut = Linear(in_channels, in_channels)


    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


def hook_fn(grad, tensor):
    print("Gradient:", grad)
    print("Tensor:", tensor)

class GSGEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels, out_channels, num_layers,
                 encoder, use_proj_head=True, proj_head_share=True,
                 pooling='max', temperature=0.2, dropout=None):
        super(GSGEncoder, self).__init__()

        self.encoder = encoder
        self.use_proj_head = use_proj_head
        self.proj_head_share = proj_head_share
        self.pooling = pooling
        self.temperature = temperature

        num_hidden=hidden_channels #128
        num_proj_hidden=64
        tau=0.4
        self.tau=tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.embedding_dim = self.encoder.hidden_channels


        self.proj_head_g1 = Project_Head(in_channels=self.embedding_dim)
        self.proj_head_g2 = Project_Head(in_channels=self.embedding_dim)

        self.fc = Sequential(
                             Linear(self.embedding_dim, self.embedding_dim),
                             ReLU(), Dropout(p=dropout, inplace=False),
                             Linear(self.embedding_dim, out_channels),
                             )

        self.init_emb()


    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, edge_attr, device=None):
        node_reps, graph_reps = self.encoder(x,edge_index, batch, edge_attr)
        # dim=2
        pred_out = self.fc(graph_reps)
        if self.use_proj_head:
            graph_reps = self.proj_head_g1(graph_reps)
        return node_reps, graph_reps, pred_out

    def loss_su(self, pred_out, target):
        loss = torch.nn.CrossEntropyLoss()
        return loss(pred_out, target)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        # z.register_hook(lambda grad: hook_fn(grad, z))
        z = F.elu(self.fc1(z))
        # z.register_hook(lambda grad: hook_fn(grad, z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        # z1.register_hook(lambda grad: hook_fn(grad, z1))
        z2 = F.normalize(z2)
        # z2.register_hook(lambda grad: hook_fn(grad, z2))
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def loss_un(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def loss_cal(self, x, x_aug, pred_out, target, Lambda):
        loss_un = self.loss_un(x, x_aug)
        loss_su = self.loss_su(pred_out, target)
        # loss_su = self.loss()
        return loss_su + Lambda * loss_un, loss_un, loss_su
