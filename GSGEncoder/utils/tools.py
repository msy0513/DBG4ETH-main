#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def setup_seed(seed):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def data_split(X, Y, seeds, K):
 
    train_splits = []
    test_splits = []
    val_splits = []

    for seed in seeds:
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        for train_val_idx, test_idx in kf.split(X=X, y=Y):
            kf_val = StratifiedKFold(n_splits=K-1, shuffle=True, random_state=seed)
            x = X[train_val_idx]
            y = Y[train_val_idx]
            for train_idx, val_idx in kf_val.split(X=x, y=y):
                test_splits.append(X[test_idx].tolist())
                train_splits.append(x[train_idx].tolist())
                val_splits.append(x[val_idx].tolist())


    for i, train_idx in enumerate(train_splits):
        assert set(train_idx + test_splits[i] + val_splits[i]) == set(X.tolist())

    print("train_idx" + str(train_splits))
    print("test_splits" + str(test_splits))
    print("val_splits" + str(val_splits))


    return train_splits, val_splits, test_splits



def my_data_split(X, Y, seeds, train_size=0.6, val_size=0.1):
    train_splits = []
    test_splits = []
    val_splits=[]

    for seed in seeds:
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=seed)

        X_train, X_remaining, Y_train, Y_remaining = train_test_split(X, Y, train_size=train_size, random_state=seed)


        X_val, X_test, Y_val, Y_test = train_test_split(X_remaining, Y_remaining, test_size=val_size / (1 - train_size),
                                                        random_state=seed)
        train_splits.append(X_train.tolist())
        val_splits.append(X_val.tolist())
        test_splits.append(X_test.tolist())

    for i, train_idx in enumerate(train_splits):
        assert set(train_idx + val_splits[i] + test_splits[i]) == set(X.tolist())

    print("train_idx:" + str(train_splits))
    print("test_idx:" + str(test_splits))
    print("val_idx:" + str(val_splits))
    return train_splits, val_splits, test_splits

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_results = None

    def __call__(self, val_loss, results):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_results = results
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
            # save best result
            self.best_results = results

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"     INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('     INFO: Early stopping')
                self.early_stop = True
