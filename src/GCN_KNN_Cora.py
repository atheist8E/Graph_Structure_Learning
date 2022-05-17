import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.datasets import Planetoid
from torch_geometric.seed import seed_everything
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

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
	parser.add_argument("--num_epochs",    		type = int,        default = 200)
	parser.add_argument("--milestone_0",   		type = int,        default = 200)
	parser.add_argument("--milestone_1",   		type = int,        default = 200)
	parser.add_argument("--hidden_channels",    type = int,        default = 16)
	parser.add_argument("--k",   				type = int,        default = 6)
	parser.add_argument("--learning_rate", 		type = float,      default = 0.01)
	parser.add_argument("--gpu",           		type = int,        default = 2)
	parser.add_argument("--random_seed",        type = int,        default = 0)
	parser.add_argument("--model",              type = str,        default = "GCN_KNN")
	parser.add_argument("--description",        type = str,        default = "GCN KNN Graph")
	return parser.parse_args()


if __name__ == "__main__":

	args = set_args()

	seed_everything(args.random_seed)

	print("source_path: {}".format(args.source_path))
	print("target_path: {}".format(args.target_path))
	print("dataset: {}".format(args.dataset))
	print("num_epochs: {}".format(args.num_epochs))
	print("milestone_0: {}".format(args.milestone_0))
	print("milestone_1: {}".format(args.milestone_1))
	print("hidden_channels: {}".format(args.hidden_channels))
	print("k: {}".format(args.k))
	print("learning_rate: {}".format(args.learning_rate))
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

		#################################################### Data Loader ###################################################

		dataset = Planetoid(root = os.path.join(args.source_path, args.dataset + "_KNN_{}".format(args.k)), name = "Cora",  pre_transform = Compose([KNN(k = args.k)]))

		################################################## Model Training ##################################################

		args.num_classes = dataset.num_classes
		args.num_node_features = dataset.num_node_features
		model = GCN_NC(args, writer).cuda(args.gpu)
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 5e-4)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.milestone_0, args.milestone_1] , gamma = 0.1)
		model.fit(dataset, criterion, optimizer, scheduler, args.num_epochs)

		################################################## Model Testing ###################################################

		max_accuracy = model.max_accuracy
		print("max_accuracy: {}".format(model.max_accuracy))

		############################################### Experiment Recording ###############################################
		
		end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		f.write("start_time: {}\n".format(start_time))
		f.write("end_time: {}\n".format(end_time))
		f.write("source_path: {}\n".format(args.source_path))
		f.write("target_path: {}\n".format(args.target_path))
		f.write("dataset: {}\n".format(args.dataset))
		f.write("num_epochs: {}\n".format(args.num_epochs))
		f.write("milestone_0: {}\n".format(args.milestone_0))
		f.write("milestone_1: {}\n".format(args.milestone_1))
		f.write("hidden_channels: {}\n".format(args.hidden_channels))
		f.write("k: {}\n".format(args.k))
		f.write("learning_rate: {}\n".format(args.learning_rate))
		f.write("gpu: {}\n".format(args.gpu))
		f.write("random_seed: {}\n".format(args.random_seed))
		f.write("model: {}\n".format(args.model))
		f.write("description: {}\n".format(args.description))
		f.write("max_accuracy: {}\n".format(max_accuracy))
		torch.save(model.state_dict(), os.path.join(args.result_path, "model.pth"))
