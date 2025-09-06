import os
import numpy as np
from math import sqrt
from scipy import stats
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

import torch_geometric
from torch_geometric.data import data as pyg_data
from torch_geometric.data import storage as pyg_storage

# Allow safe unpickling of custom PyG classes in torch >= 2.6
torch.serialization.add_safe_globals([
    pyg_data.DataEdgeAttr,
    pyg_data.DataTensorAttr,
    pyg_storage.GlobalStorage,
])

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='data', dataset='davis',
                 xd=None, xt=None, y=None,
                 transform=None, pre_transform=None,
                 smile_graph=None):
        self.dataset = dataset  # e.g., 'kiba_train'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print(f'Pre-processed data found: {self.processed_paths[0]}, loading ...')
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print(f'Pre-processed data {self.processed_paths[0]} not found, doing pre-processing...')
            self._custom_process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']  # e.g., kiba_train.pt

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass  # unused, handled by _custom_process()

    def _custom_process(self, xd, xt, y, smile_graph):
        assert len(xd) == len(xt) == len(y), "Lengths mismatch!"
        data_list = []
        for i in range(len(xd)):
            print(f'Converting SMILES to graph: {i+1}/{len(xd)}')
            smiles = xd[i]
            target = xt[i]
            label = y[i]

            c_size, features, edge_index = smile_graph[smiles]
            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).transpose(1, 0),
                y=torch.tensor([label], dtype=torch.float)
            )
            data.xt = torch.tensor(target, dtype=torch.float)
            data.c_size = torch.tensor([c_size], dtype=torch.long)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# ===== Evaluation Metrics =====

def rmse(y, f):
    return sqrt(((y - f) ** 2).mean())

def mse(y, f):
    return ((y - f) ** 2).mean()

def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]

def spearman(y, f):
    return stats.spearmanr(y, f)[0]

def ci(y, f):
    ind = np.argsort(y)
    y, f = y[ind], f[ind]
    n = len(y)
    S = 0.0
    z = 0.0
    for i in range(n):
        for j in range(i):
            if y[i] > y[j]:
                z += 1
                if f[i] > f[j]:
                    S += 1
                elif f[i] == f[j]:
                    S += 0.5
    return S / z if z != 0 else 0.0
