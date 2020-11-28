import os
import torch as t
import numpy as np
from torch.utils.data import TensorDataset
from torch.distributions import Normal

__all__ = ('UCI',)


class Dataset:
    """
    Represents the full dataset.  We will have two copies: one normalised, one unnormalized.
    """
    def __init__(self, X, y, index_train, index_test):
        self.X = X
        self.y = y

        self.train_X = self.X[index_train]
        self.train_y = self.y[index_train]
        self.test_X  = self.X[index_test]
        self.test_y  = self.y[index_test]


class UCI:
    """
    The usage is:
    ```
    uci = UCIDataset("protein", 3)
    ```
    e.g. normalized training dataset:
    ```
    uci.norm.train
    ```
    """
    def __init__(self, dataset, split, mbatch_size, dtype='float32'):
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        dataset_dir = f'{_ROOT}/{dataset}/'
        data = np.loadtxt(f'{dataset_dir}/data.txt').astype(getattr(np, dtype))
        index_features = np.loadtxt(f'{dataset_dir}/index_features.txt')
        index_target = np.loadtxt(f'{dataset_dir}/index_target.txt')
        unnorm_X = t.from_numpy(data[:, index_features.astype(int)])
        unnorm_y = t.from_numpy(data[:, index_target.astype(int):index_target.astype(int)+1])

        # split into train and test
        index_train = np.loadtxt(f'{dataset_dir}/index_train_{split}.txt').astype(int)
        index_test  = np.loadtxt(f'{dataset_dir}/index_test_{split}.txt').astype(int)

        unnorm_train_X = unnorm_X[index_train]
        unnorm_train_y = unnorm_y[index_train]

        # compute normalization constants based on training set
        self.X_std = t.std(unnorm_train_X, 0)
        self.X_std[self.X_std == 0] = 1. # ensure we don't divide by zero
        self.X_mean = t.mean(unnorm_train_X, 0)

        self.y_mean = t.mean(unnorm_train_y)
        self.y_std  = t.std(unnorm_train_y)

        X_norm = (unnorm_X - self.X_mean)/self.X_std
        y_norm = (unnorm_y - self.y_mean)/self.y_std

        self.trainset = TensorDataset(X_norm[index_train], y_norm[index_train])
        self.testset  = TensorDataset(X_norm[index_test],  y_norm[index_test])

        self.trainloader = t.utils.data.DataLoader(
            self.trainset, 
            batch_size=mbatch_size, 
            shuffle=False, #True, 
            num_workers=0
        )
        self.testloader = t.utils.data.DataLoader(
            self.testset, 
            batch_size=mbatch_size, 
            shuffle=False, 
            num_workers=0
        )

        self.num_train_set = unnorm_train_X.shape[0]
        self.in_features   = unnorm_train_X.shape[1]
        self.out_features  = unnorm_train_y.shape[1]


    def denormalize_y(self, y):
        return self.y_std * y + self.y_mean

    def denormalize_Py(self, Py):
        return Normal(self.y_mean + self.y_std*Py.loc, self.y_std*Py.scale)


