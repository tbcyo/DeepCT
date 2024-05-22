import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def split(dat, t_grid = None, inf_ratio = 0.3):
    n_sample = len(t_grid[0])
    n_train = round(n_sample * (1-inf_ratio))

    perm = np.random.permutation(n_sample)
    dat = dat[perm, :]
    t_grid = t_grid[:, perm]
    training = dat[:n_train, :]
    testing = dat[(n_train+1):, :]
    t_grid_test = t_grid[:, (n_train+1):]
    return torch.from_numpy(training).float(), torch.from_numpy(testing).float(), torch.from_numpy(t_grid_test).float()

def split0(dat, inf_ratio = 0.3):
    n_sample = len(dat.shape[1])
    n_train = round(n_sample * (1-inf_ratio))

    perm = np.random.permutation(n_sample)
    dat = dat[perm, :]
    training = dat[:n_train, :]
    testing = dat[(n_train+1):, :]
    return torch.from_numpy(training).float(), torch.from_numpy(testing).float()


def kfold0(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def kfold(dat, t_grid, inf_ratio):
    n_sample = len(t_grid[0])
    k = int(1/inf_ratio)

    perm = np.random.permutation(n_sample).tolist()
    out = kfold0(perm, k)

    new_dat = []
    for i in range(k):
        train_idx = []
        test_idx = out[i]
        for j in range(k):
            if i != j:
                train_idx += out[j]
        training = dat[train_idx, :]
        testing = dat[test_idx, :]
        t_grid_test = t_grid[:, test_idx]
        new_dat.append((torch.from_numpy(training).float(), torch.from_numpy(testing).float(), torch.from_numpy(t_grid_test).float()))
    return new_dat
