import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

from torch.autograd import Variable
import numpy as np


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj, dynamic=False):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            if dynamic:
                y = F.normalize(y, p=2, dim=1)
            else:
                y = F.normalize(y, p=2, dim=2)
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0, dynamic=False):
        if dynamic:
            conv_first = GraphConv(input_dim=embedding_dim * 3, output_dim=hidden_dim, add_self=add_self,
                                   normalize_embedding=normalize, bias=self.bias)
            conv_first_0 = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                                     normalize_embedding=normalize, bias=self.bias)
        else:
            conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                                   normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        if dynamic:
            return conv_first, conv_first_0, conv_block, conv_last
        else:
            return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None, concat=False, time_0=False,
                    dynamic=False, conv_fir_0=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        if time_0:
            x = conv_fir_0(x, adj, dynamic)
        else:
            x = conv_first(x, adj, dynamic)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x, adj, dynamic)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj, dynamic)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        if concat:
            if dynamic:
                x_tensor = torch.cat(x_all, dim=1)
            else:
                x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            # print("pred"+str(pred))
            # print("label"+str(label))
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.
    """
    def __init__(self, input_size, hidden_size, x_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.h2h_0 = nn.Linear(x_dim, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, t=False):
        # print(x.size())
        gate_x = self.x2h(x)

        if t:
            gate_h = self.h2h_0(hidden)
        else:
            gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)

        inputgate = F.sigmoid(i_i + h_i)

        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class diffpool(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1, pool_method='diffpool',
                 pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers
            before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(diffpool, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                       num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                       args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.max_num_nodes = max_num_nodes
        self.pool_method = pool_method
        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        self.conv_fir, self.conv_fir_1, self.conv_blo, self.conv_las = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout, dynamic=True)
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)

            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        self.atten_time = torch.nn.Parameter(torch.ones(10).cuda(), requires_grad=True)


        # self.atten_time.weight.data = init.xavier_uniform(self.atten_time.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.uniform_(self.atten_time)#self.atten_time.data.fill_(1)

        self.gru_1 = GRUCell(self.max_num_nodes, embedding_dim * 3, label_dim, True)
        self.gru_2 = GRUCell(int(max_num_nodes * assign_ratio), embedding_dim * 3, label_dim, True)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, Is_attack=False, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        if Is_attack == False:
            A = torch.cat([torch.unsqueeze(torch.cat([torch.unsqueeze(torch.from_numpy(adj[i][j].A),0) 
                for j in range(len(adj[0]))],0),0)  for i in range(len(adj))],0).cuda()
        else:
            A = adj
        a=A.size()

        X = torch.cat([torch.unsqueeze(
            torch.cat([torch.unsqueeze(
                torch.from_numpy(x[i][j].A), 0) for j in range(len(x[0]))], 0), 0) for i in range(len(x))], 0).cuda()

        max_num_nodes = A.shape[2]
        embedding_m = []
        if batch_num_nodes is not None:
            for i in range(len(batch_num_nodes)):
                # for i in range(batch_num_nodes.shape[0]):
                embedding_m.append(self.construct_mask(max_num_nodes, batch_num_nodes[i]))
                embedding_mask = torch.cat([torch.unsqueeze(embedding_m[i], 0) for i in range(len(embedding_m))], 0)
        else:
            embedding_mask = None

        out_all = []

        # diff-GRU_0
        self.embedding_GRU = self.diff_GRU(A, X, self.conv_fir, self.conv_blo, self.conv_las, self.gru_1,
                                           embedding_mask, conv_fir_1=self.conv_fir_1, ifmask=True)
        time_embedding_ = []

        for i in range(self.embedding_GRU.shape[0]):
            # self.embedding_GRU_pool_1.append([])
            t_e_ = []

            for j in range(self.embedding_GRU.shape[0]):
                # print(self.embedding_GRU[i][j])
                t_e_.append(self.embedding_GRU[i][j] * self.atten_time[j])
            time_embedding_.append(torch.cat([torch.unsqueeze(t_e_[i], 0) for i in range(len(t_e_))], 0))
            
        self.embedding_GRU_1 = torch.cat(
            [torch.unsqueeze(time_embedding_[i], 0) for i in range(len(time_embedding_))], 0)

        out_sum = torch.sum(self.embedding_GRU_1, dim=1)
        out, _ = torch.max(out_sum, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(out_sum, dim=1)
            out_all.append(out)

        for n_p in range(self.num_pooling):
            embedding_m = []
            if batch_num_nodes is not None and n_p == 0:
                for i in range(len(batch_num_nodes)):
                    embedding_m.append(self.construct_mask(max_num_nodes, batch_num_nodes[i]))
                    embedding_mask = torch.cat([torch.unsqueeze(embedding_m[i], 0) for i in range(len(embedding_m))], 0)
            else:
                embedding_mask = None

            self.assign_tensor = self.dynamic_gcn(A, X, self.assign_conv_first_modules[n_p],
                                                  self.assign_conv_block_modules[n_p],
                                                  self.assign_conv_last_modules[n_p], embedding_mask)
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[n_p](self.assign_tensor))

            self.assign_tensor_all = []

            if embedding_mask is not None:
                x = []
                adj = []
                for i in range(embedding_mask.shape[0]):
                    self.assign_tensor_all.append(self.assign_tensor * embedding_mask[i])
                    # update pooled features and adj matrix
                    x.append(torch.matmul(torch.transpose(self.assign_tensor_all[i], 1, 2), self.embedding_GRU[i]))
                    adj.append(torch.transpose(self.assign_tensor_all[i], 1, 2) @ A[i] @ self.assign_tensor_all[i])
                    x_a = x
                A_pool = torch.cat([torch.unsqueeze(adj[i], 0) for i in range(len(adj))], 0)
                X_pool = torch.cat([torch.unsqueeze(x[i], 0) for i in range(len(x))], 0)


            self.embedding_GRU_pool = self.diff_GRU(A_pool, X_pool, self.conv_first_after_pool[n_p],
                                                    self.conv_block_after_pool[n_p],
                                                    self.conv_last_after_pool[n_p],
                                                    self.gru_2, embedding_mask,
                                                    conv_fir_1=self.conv_first_after_pool[n_p],
                                                    ifmask=False)

            
            time_embedding = []
            for i in range(self.embedding_GRU_pool.shape[0]):
                # self.embedding_GRU_pool_1.append([])
                t_e = []
                for j in range(self.embedding_GRU_pool.shape[0]):
                    # print(self.embedding_GRU[i][j])
                    t_e.append(self.embedding_GRU_pool[i][j] * self.atten_time[j])
                time_embedding.append(torch.cat([torch.unsqueeze(t_e[i], 0) for i in range(len(t_e))], 0))
                
            self.embedding_GRU_pool_1 = torch.cat([torch.unsqueeze(time_embedding[i], 0) for i in range(len(time_embedding))], 0)
            
            out_sum = torch.sum(self.embedding_GRU_pool_1, dim=1)
            out, _ = torch.max(out_sum, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(out_sum, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        # torch.save(output, 'output.pt')
        ypred = self.pred_model(output)

        return ypred, self.atten_time,output

    def diff_GRU(self, A, X, conv_fir, conv_blo, conv_las, gru, embedding_mask, conv_fir_1=None, ifmask=False):
        embedding_GRU = []
        for g_index in range(A.shape[0]):
            em_gcn = []
            em_gru = []
            for time in range(A.shape[1]):
                if ifmask:
                    if time == 0:
                        em_gcn.append(self.gcn_forward(X[g_index][0], A[g_index][time],
                                                       conv_fir, conv_blo, conv_las, embedding_mask[g_index][time],
                                                       concat=True, time_0=True, dynamic=True, conv_fir_0=conv_fir_1))
                    else:
                        em_gcn.append(self.gcn_forward(em_gru[-1], A[g_index][time],
                                                       conv_fir, conv_blo, conv_las,
                                                       embedding_mask[g_index][time], concat=True, time_0=False,
                                                       dynamic=True, conv_fir_0=conv_fir_1))
                else:
                    if time == 0:
                        em_gcn.append(self.gcn_forward(X[g_index][0], A[g_index][time],
                                                       conv_fir, conv_blo, conv_las, concat=True, time_0=True,
                                                       dynamic=True, conv_fir_0=conv_fir_1))
                    else:
                        em_gcn.append(self.gcn_forward(em_gru[-1], A[g_index][time],
                                                       conv_fir, conv_blo, conv_las,
                                                       concat=True, time_0=False,
                                                       dynamic=True, conv_fir_0=conv_fir_1))
                em_gru.append(gru(A[g_index][time], em_gcn[time], t=False))
            embedding_GRU.append(em_gru)
            embedding = torch.cat([torch.unsqueeze(
                        torch.cat([torch.unsqueeze(
                            embedding_GRU[i][j], 0) for j in range(len(embedding_GRU[0]))], 0), 0) 
                            for i in range(len(embedding_GRU))], 0).cuda()
            # embedding = torch.cat()
        return embedding

    def dynamic_gcn(self, A, X, conv_fir, conv_blo, conv_las, embedding_mask):
        for g_index in range(A.shape[0]):
            em_gcn = []
            for time in range(A.shape[1]):
                em_gcn.append(self.gcn_forward(X[g_index][time], A[g_index][time],
                                               conv_fir, conv_blo, conv_las,
                                               embedding_mask[g_index][time], concat=True, time_0=False,
                                               dynamic=True))

        embedding = torch.cat([torch.unsqueeze(em_gcn[j], 0) for j in range(len(em_gcn))])

        return embedding

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(diffpool, self).loss(pred, label)

        return loss
