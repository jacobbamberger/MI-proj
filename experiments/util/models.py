import torch
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax
import torch_geometric.nn as nn
import torch.nn.functional as F

import time


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GnnBaseline(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.gin_input = self.get_GIN(dataset.num_node_features, 16, 16)
        self.gin16 = self.get_GIN(16, 16, 16)

        # self.pool = nn.EdgePooling(16) $ this layer could not run on GPU for Jacob
        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 + 1, 16),
            ELU(alpha=0.1),
            Dropout(p=0.5),
            Linear(16, dataset.num_classes),
            Softmax(dim=1)
        )

    @staticmethod
    def get_GIN(in_dim, h_dim, out_dim):
        MLP = Sequential(
            Linear(in_dim, h_dim),
            BatchNorm1d(h_dim),
            ReLU(),
            Linear(h_dim, out_dim)
        )
        return nn.GINConv(MLP, eps=0.0, train_eps=False)


class NoPhysicsGnn(GnnBaseline):
    """Model for WssToCnc and CoordToCnc"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def forward(self, x, edge_index, batch, segment):

        x = self.gin_input(x, edge_index)
        x = F.elu(x, alpha=0.1)
        # x, edge_index, batch, _ = self.pool(x, edge_index, batch)
        #print(x.shape)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        # x, edge_index, batch, _ = self.pool(x, edge_index, batch)
        #print(x.shape)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        x1 = self.gmp(x, batch)
        x2 = self.gap(x, batch)
        #print(x.shape)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)
        x = self.classifier(x)
        #print(x.shape)
        return x