import os
import torch
import pickle
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


class CIFAR_Graph_Dataset(Data):
    def __init__(self, args, root, train_fname, test_fname):
        super().__init__()
        self.args = args

        self.train_path = os.path.join(root, train_fname)
        with open(self.train_path, "rb") as f:
            train_data = pickle.load(f)
            self.train_x = torch.tensor(train_data["feats"], dtype = torch.float)
            self.train_y = torch.tensor(train_data["labels"], dtype = torch.long)
            self.train_mask_ = torch.ones(self.train_y.size(0)).bool()

        self.test_path = os.path.join(root, test_fname)
        with open(self.test_path, "rb") as f:
            test_data = pickle.load(f)
            self.test_x = torch.tensor(test_data["feats"], dtype = torch.float)
            self.test_y = torch.tensor(test_data["labels"], dtype = torch.long)
            self.test_mask_ = torch.ones(self.test_y.size(0)).bool()
            
    @property
    def x(self):
        return torch.cat([self.train_x, self.test_x], dim = 0).cuda(self.args.gpu)

    @property
    def y(self):
        return torch.cat([self.train_y, self.test_y], dim = 0).cuda(self.args.gpu)

    @property
    def num_nodes(self):
        return self.x.size(0)

    @property
    def num_classes(self):
        return torch.unique(self.y).numel()

    @property
    def num_node_features(self):
        return self.x.size(1)

    @property
    def train_mask(self):
        return torch.cat([self.train_mask_, ~self.test_mask_], dim = 0).cuda(self.args.gpu)

    @property
    def test_mask(self):
        return torch.cat([~self.train_mask_, self.test_mask_], dim = 0).cuda(self.args.gpu)

    @property
    def edge_index(self):
        return knn_graph(x = self.x.cuda(self.args.gpu), k = self.args.k, loop = False)

class Credit_Graph_Dataset(Data):
    def __init__(self, args, root, train_fname, test_fname):
        super().__init__()
        self.args = args

        self.train_path = os.path.join(root, train_fname)
        with open(self.train_path, "rb") as f:
            train_data = pickle.load(f)
            self.train_x = torch.tensor(train_data["feats"], dtype = torch.float)
            self.train_y = torch.tensor(train_data["labels"], dtype = torch.long)
            self.train_mask_ = torch.ones(self.train_y.size(0)).bool()

        self.test_path = os.path.join(root, test_fname)
        with open(self.test_path, "rb") as f:
            test_data = pickle.load(f)
            self.test_x = torch.tensor(test_data["feats"], dtype = torch.float)
            self.test_y = torch.tensor(test_data["labels"], dtype = torch.long)
            self.test_mask_ = torch.ones(self.test_y.size(0)).bool()
            
    @property
    def x(self):
        return torch.cat([self.train_x, self.test_x], dim = 0).cuda(self.args.gpu)

    @property
    def y(self):
        return torch.cat([self.train_y, self.test_y], dim = 0).cuda(self.args.gpu)

    @property
    def num_nodes(self):
        return self.x.size(0)

    @property
    def num_classes(self):
        return torch.unique(self.y).numel()

    @property
    def num_node_features(self):
        return self.x.size(1)

    @property
    def train_mask(self):
        return torch.cat([self.train_mask_, ~self.test_mask_], dim = 0).cuda(self.args.gpu)

    @property
    def test_mask(self):
        return torch.cat([~self.train_mask_, self.test_mask_], dim = 0).cuda(self.args.gpu)

    @property
    def edge_index(self):
        return knn_graph(x = self.x.cuda(self.args.gpu), k = self.args.k, loop = False)
