import os
import torch
import random
import pandas as pd
import torch.nn as nn
import networkx as nx
import seaborn as sns
import torch.nn.init as init
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from matplotlib.patches import Patch
from torch_geometric.data import Data
import torch_geometric.nn as graph_nn
from torch_geometric.utils import to_networkx, to_dense_batch, to_dense_adj, dense_to_sparse, coalesce, subgraph, k_hop_subgraph, negative_sampling, add_self_loops, remove_self_loops 

class Confusion_GSL(nn.Module):
    def __init__(self, args, writer):
        super().__init__()
        self.args = args
        self.writer = writer
        self.max_accuracy = 0
        self.conv1 = graph_nn.GCNConv(args.num_features, 16)
        self.relu1 = nn.ReLU()
        self.conv2 = graph_nn.GCNConv(16, args.num_classes)

    def forward(self, G):
        out = self.conv1(G.x, G.edge_index)
        out = self.relu1(out)
        out = self.conv2(out, G.edge_index)
        return out

    def fit(self, G, criterion, optimizer, scheduler, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            with torch.enable_grad():
                self.train()
                outputs = self.forward(G)
                P = F.softmax(outputs, dim = 1)
                P_max, preds = P.max(dim = 1)
                A_star = torch.matmul(P, P.T)
                loss = criterion(outputs[G.train_mask], G.y[G.train_mask]) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                correct = (preds[G.test_mask] == G.y[G.test_mask]).sum().item()
                accuracy = float(correct) / int(G.test_mask.sum())
                (self.writer).add_scalar("Loss | {}".format(self.args.model_name), loss.item(), epoch)
                (self.writer).add_scalar("Accuracy | {}".format(self.args.model_name), accuracy, epoch)
                (self.writer).add_scalar("Max Accuracy | {}".format(self.args.model_name), self.max_accuracy, epoch)
                if accuracy > self.max_accuracy:
                    self.max_accuracy = accuracy
                scheduler.step()
        return A_star

    def visualize(self, G, num_hops = 1):
        self.eval()
        outputs = self.forward(G)
        P = F.softmax(outputs, dim = 1)
        P_label = torch.gather(P, dim = 1, index = G.y.unsqueeze(axis = 1))
        lmap = {0: "Petit", 1: "Train", 2: "Val", 3: "Test"}
        node_mask = G.train_mask * 1 + G.val_mask * 2 + G.test_mask * 3
        cmap = {0: "blue", 1: "green", 2: "red", 3: "cyan", 4: "magenta", 5: "yellow", 6: "black"}
        class_patches = [Patch(facecolor = color, label = cls) for cls, color in cmap.items()]
        for i, mask in enumerate(node_mask):
            subset, edge_index, _, _ = k_hop_subgraph(node_idx = i, num_hops = num_hops, edge_index = G.edge_index, relabel_nodes = True)
            edge_index_original = torch.vstack((torch.gather(subset, dim = 0, index = edge_index[0]), torch.gather(subset, dim = 0, index = edge_index[1])))
            node_color = [cmap[label] for label in G.y[subset].tolist()]
            labels = {j: "{}\n".format(node_idx) + lmap[mask.item()] + "\n{:.2f}".format(prob.item()) \
                      for j, (node_idx, mask, prob) in enumerate(zip(subset, node_mask[subset], P_label[subset]))}
            sub_graph = Data(x = G.x[subset], edge_index = edge_index, y = G.y[subset])
            g = to_networkx(sub_graph, node_attrs = ["y"])
            pos = nx.spring_layout(g)
            nx.draw(g, pos, node_color = node_color, arrows = False, labels = labels)
            plt.legend(handles = class_patches, loc = "lower right")
            plt.savefig(os.path.join(self.args.result_path, "{}.png".format(i)))
            plt.close()
