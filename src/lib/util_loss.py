import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_dense_adj, dense_to_sparse, subgraph

class FeasibilityLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.eps = 1e-15
        self.args = args

    def forward(self, A_star, edge_index, node_mask):
        A = to_dense_adj(edge_index, max_num_nodes = self.args.num_nodes)[0]
        A_mask = 1 - (torch.matmul(node_mask.float().unsqueeze(1), node_mask.float().unsqueeze(0)) + torch.matmul((~node_mask).float().unsqueeze(1), (~node_mask).float().unsqueeze(0)))
        pos_edge_index, _ = dense_to_sparse(A * A_mask)
        pos_loss = (-1 * torch.log(nn.Sigmoid()((A_star[pos_edge_index[0]] * A_star[pos_edge_index[1]]).sum(dim = 1)) + self.eps)).sum()
        return pos_loss
