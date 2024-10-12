import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import time
from data0306 import *


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method +'_' + str(args.num_nodes) +'_' +str(args.max_links)
    return name


def train_phishing_detector_dy(train_dataset, model, train_num, val_num, test_num, args, val_dataset=None, test_dataset=None, writer=None):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0
        }
    best_test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0
        }
    train_accs = []
    train_epochs = []
    val_accs = []
    best_val_epochs = []
    test_accs = []
    best_test_epochs = []
    all_time = 0
    # print("===============================================")
    # print(range(train_num))
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        t = time.time()
        train_pred_label_list=[]
        print('Epoch: ', epoch)
        for batch_idx in range(train_num):
            begin_time = time.time()
            model.zero_grad()
            # if args.normalize:
            #     adj = Variable(Adj[0].float(), requires_grad=False)
            # else:
            # print(batch_idx)
            adj = train_dataset['Adj'][batch_idx]
            h0 = train_dataset['Fea'][batch_idx]
            label = (Variable(torch.Tensor(train_dataset['Label'][batch_idx])).cuda()).to(torch.int64)
            batch_num_nodes = train_dataset['Batch_num_nodes'][batch_idx]
            assign_input = train_dataset['Fea'][batch_idx]
            ypred, att,_ = model(h0, adj, batch_num_nodes, assign_x=assign_input)

            train_pred_label_list.append(ypred)
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            iter += 1
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed

        train_pred = torch.cat(train_pred_label_list).cpu().detach().numpy()

        train_pred_transposed = {f'col{i + 1}': train_pred[:, i].tolist() for i in range(train_pred.shape[1])}
        df = pd.DataFrame(train_pred_transposed)

        print("write to train_result succeed ")


        train_time = time.time() - t
        all_time += train_time
        # avg_loss /= batch_idx + 1
        avg_loss /= train_num
        # avg_loss /= train_num + 0.001

        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)

        # print(att)
        print('Avg loss: ', avg_loss, 'epoch time: ', total_time, 'train_time: ', all_time/(epoch+1))
        result = evaluate_dynamic(train_dataset, train_num, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)

        if val_dataset is not None:
            val_result = evaluate_dynamic(val_dataset, val_num, model, args, name='Validation')
            val_accs.append(val_result['acc'])
            if val_result['acc'] > best_val_result['acc'] - 1e-7:
                best_val_result['acc'] = val_result['acc']
                best_val_result['epoch'] = epoch
                best_val_result['loss'] = avg_loss
                best_val_result['pre'] = val_result['prec']
                best_val_result['recall'] = val_result['recall']
                best_val_result['F1'] = val_result['F1']

                if val_result['acc'] > best_val_result['acc'] - 1e-7:
                    best_model_state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(best_model_state, 'D:/2-code/tegdetector/model_param/TEGD.pth')

            print('Best val result: ', best_val_result)
            best_val_epochs.append(best_val_result['epoch'])


        # if test_result['acc'] > best_test_result['acc'] - 1e-7:
        #     model_state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        #     mth = os.path.join('D:/2-code/tegdetector/model_param/TEGD.pth')
        #     torch.save(model_state, mth)
    if test_dataset is not None:
        test_result = evaluate_dynamic(test_dataset, test_num, model, args, name='Test')
        test_accs.append(test_result['acc'])
        if test_result['acc'] > best_test_result['acc'] - 1e-7:
            best_test_result['acc'] = test_result['acc']
            best_test_result['epoch'] = epoch
            best_test_result['loss'] = avg_loss
            best_test_result['pre'] = test_result['prec']
            best_test_result['recall'] = test_result['recall']
            best_test_result['F1'] = test_result['F1']
        print('Best Test result: ', best_test_result)
        best_test_epochs.append(best_test_result['epoch'])
    print(all_time / args.num_epochs)
    return  model, val_accs


def evaluate_dynamic(dataset, batch_num, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    graph=[]
    # print(f'dataset:{dataset}')
    # print(batch_num)
    for batch_idx in range(batch_num):#range(batch_num)
        adj = dataset['Adj'][batch_idx]
        h0 = dataset['Fea'][batch_idx]
        # labels_batch = (Variable(torch.Tensor(dataset['Label'][batch_idx])).cuda()).to(torch.int64)
        labels.append((Variable(torch.Tensor(dataset['Label'][batch_idx])).cuda()).to(torch.int64))
        batch_num_nodes = dataset['Batch_num_nodes'][batch_idx]
        assign_input = dataset['Fea'][batch_idx]


        num_samples = adj.shape[0]

        ypred, att ,graph_pre= model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        graph.append(graph_pre.cpu().data.numpy())


    # labels = np.hstack(labels)
    labels = np.hstack(np.array([i.cpu().numpy() for i in labels]))
    preds = np.hstack(preds)
    graph=np.hstack(graph)



    result = {'pred_result':preds,
              'prec': metrics.precision_score(labels, preds, average='macro', zero_division=0),
              'recall': metrics.recall_score(labels, preds, average='macro', zero_division=0),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro", zero_division=0)}
    df = pd.DataFrame(result)



    print(f"valortest_result:{preds}")
    print(name, "  recall",result['recall'],  " accuracy:", result['acc'], " F1:", result['F1'])
    return result
