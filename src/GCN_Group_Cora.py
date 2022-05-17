import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from pprint import pprint
from itertools import product
from datetime import datetime
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.seed import seed_everything
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose, NormalizeFeatures

from lib.util_loss import *
from lib.util_dataset import *
from lib.util_transform import *
from lib.util_architecture import *

def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",        type = str,        default = "../dat/")
	parser.add_argument("--source_file",        type = str,        default = "")
	parser.add_argument("--target_path",        type = str,        default = "../log/")
	parser.add_argument("--target_file",        type = str,        default = "")
	parser.add_argument("--dataset",       		type = str,        default = "Cora")
	parser.add_argument("--num_epochs_0",    	type = int,        default = 200)
	parser.add_argument("--num_epochs_1",    	type = int,        default = 200)
	parser.add_argument("--learning_rate_0", 	type = float,      default = 0.005)
	parser.add_argument("--learning_rate_1", 	type = float,      default = 0.001)
	parser.add_argument("--gpu",           		type = int,        default = 2)
	parser.add_argument("--random_seed",        type = int,        default = 0)
	parser.add_argument("--model",              type = str,        default = "GCN_GSL")
	parser.add_argument("--description",        type = str,        default = "GCN GSL Group")
	return parser.parse_args()


if __name__ == "__main__":

	args = set_args()
	seed_everything(args.random_seed)

	print("source_path: {}".format(args.source_path))
	print("target_path: {}".format(args.target_path))
	print("dataset: {}".format(args.dataset))
	print("num_epochs_0: {}".format(args.num_epochs_0))
	print("num_epochs_1: {}".format(args.num_epochs_1))
	print("learning_rate_0: {}".format(args.learning_rate_0))
	print("learning_rate_1: {}".format(args.learning_rate_1))
	print("gpu: {}".format(args.gpu))
	print("random_seed: {}".format(args.random_seed))
	print("model: {}".format(args.model))
	print("description: {}".format(args.description))

	start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	args.model_path = os.path.join(args.target_path, args.model)
	if os.path.exists(args.model_path) is False:
		os.mkdir(args.model_path)
	args.result_path = os.path.join(args.model_path, start_time)
	writer = SummaryWriter(log_dir = args.result_path)
	with open(os.path.join(args.result_path, "Experiment_Description.txt"), "w") as f:

		######################################################### Preparation ############################################################

		dataset = Planetoid(root = os.path.join(args.source_path, args.dataset + "_Normalized"), name = "Cora", pre_transform = Compose([NormalizeFeatures()]))
		args.num_nodes = dataset[0].x.size(0)
		args.num_classes = dataset.num_classes
		args.num_node_features = dataset.num_node_features
		model = GCN_NC_GSL(args, writer).cuda(args.gpu)
		
		################################################## Galapagos - Imaginary ##################################################

		"""
		graph = dataset[0]
		nodes_group = defaultdict(list)
		for node_i, cls in enumerate(graph.y):
			nodes_group[cls.item()].append(node_i)
		for cls, nodes in nodes_group.items():
			edges = torch.tensor([edge for edge in product(nodes, nodes)], dtype = torch.long)
			try:
				edge_index = torch.cat((edge_index, edges), dim = 0)
			except NameError:
				edge_index = edges
		graph.edge_index = edge_index.T
		dataset = [graph]
		"""

		################################################## Galapagos - Real ##################################################

		"""
		graph = dataset[0]
		for edge in graph.edge_index.T:
			if graph.y[edge[0]] == graph.y[edge[1]]:
				try:
					edge_index = torch.cat((edge_index, edge.unsqueeze(0)), dim = 0)
				except NameError:
					edge_index = edge.unsqueeze(0)
		graph.edge_index = edge_index.T
		dataset = [graph]
		"""

		################################################## Isolate ##################################################

		graph = dataset[0]
		loop_index = torch.arange(0, graph.x.size(0), dtype = torch.long)
		loop_index = loop_index.unsqueeze(0).repeat(2, 1)
		graph.edge_index = loop_index
		dataset = [graph]

		################################################## Model Training ##################################################

		print(dataset[0].edge_index.size(1))
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.classification_model.parameters(), lr = args.learning_rate_0, weight_decay = 5e-4)
		model.fit_first_phase(dataset, criterion, optimizer, args.num_epochs_0)
		print("First Phase | Max Accuracy: {}".format(model.max_accuracy))

		################################################## Experiment Recording #########################################################
		
		end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		f.write("start_time: {}\n".format(start_time))
		f.write("end_time: {}\n".format(end_time))
		f.write("source_path: {}\n".format(args.source_path))
		f.write("target_path: {}\n".format(args.target_path))
		f.write("dataset: {}\n".format(args.dataset))
		f.write("num_epochs_0: {}\n".format(args.num_epochs_0))
		f.write("num_epochs_1: {}\n".format(args.num_epochs_1))
		f.write("learning_rate_0: {}\n".format(args.learning_rate_0))
		f.write("learning_rate_1: {}\n".format(args.learning_rate_1))
		f.write("gpu: {}\n".format(args.gpu))
		f.write("random_seed: {}\n".format(args.random_seed))
		f.write("model: {}\n".format(args.model))
		f.write("description: {}\n".format(args.description))
		f.write("max_accuracy: {}\n".format(model.max_accuracy))
		torch.save(model.state_dict(), os.path.join(args.result_path, "model.pth"))
