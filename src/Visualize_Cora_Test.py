import os
import sys
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.seed import seed_everything
from torch_geometric.utils import subgraph, k_hop_subgraph, to_networkx
from torch_geometric.transforms import Compose, NormalizeFeatures


def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",        type = str,        default = "../dat/")
	parser.add_argument("--source_file",        type = str,        default = "")
	parser.add_argument("--target_path",        type = str,        default = "../log/")
	parser.add_argument("--target_file",        type = str,        default = "")
	parser.add_argument("--dataset",       		type = str,        default = "Cora")
	parser.add_argument("--random_seed",        type = int,        default = 0)
	parser.add_argument("--model",              type = str,        default = "Cora_Visualize")
	parser.add_argument("--description",        type = str,        default = "Test Node Visualize")
	return parser.parse_args()

def visualize(dataset, result_path):
	cmap = {0: "blue", 1: "green", 2: "red", 3: "cyan",	4: "magenta", 5: "yellow", 6: "black"}
	custom_lines = [Patch(facecolor = color, label = cls) for cls, color in cmap.items()]
	graph = dataset[0]
	for i, mask in enumerate(graph.test_mask):
		if mask == True:
			subset, edge_index, _, _ = k_hop_subgraph(node_idx = i, num_hops = 2, edge_index = graph.edge_index, relabel_nodes = True)
			node_color = [cmap[label] for label in graph.y[subset].tolist()]
			labels = {i: bool(mask) for i, mask in enumerate(graph.test_mask[subset])}
			sub_graph = Data(x = graph.x[subset], edge_index = edge_index, y = graph.y[subset])
			g = to_networkx(sub_graph, node_attrs = ["y"])
			pos = nx.spring_layout(g)
			nx.draw(g, pos, node_color = node_color, arrows = False, labels = labels)
			plt.legend(handles = custom_lines, loc = "lower right")
			plt.savefig(os.path.join(result_path, "{}.png".format(i)))
			plt.close()

if __name__ == "__main__":

	args = set_args()
			
	seed_everything(args.random_seed)

	print("source_path: {}".format(args.source_path))
	print("target_path: {}".format(args.target_path))
	print("dataset: {}".format(args.dataset))
	print("random_seed: {}".format(args.random_seed))
	print("model: {}".format(args.model))
	print("description: {}".format(args.description))

	start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	args.model_path = os.path.join(args.target_path, args.model)
	if os.path.exists(args.model_path) is False:
		os.mkdir(args.model_path)
	args.result_path = os.path.join(args.model_path, start_time)
	if os.path.exists(args.result_path) is False:
		os.mkdir(args.result_path)
	dataset = Planetoid(root = os.path.join(args.source_path, args.dataset), name = "Cora")
	visualize(dataset, args.result_path)
