import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
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
    parser.add_argument("--dataset",       	    type = str,        default = "Cora")
    parser.add_argument("--num_epochs_0",       type = int,        default = 200)
    parser.add_argument("--num_epochs_1",       type = int,        default = 200)
    parser.add_argument("--milestone_0",        type = int,        default = 200)
    parser.add_argument("--milestone_1",        type = int,        default = 200)
    parser.add_argument("--learning_rate_0",    type = float,      default = 0.01)
    parser.add_argument("--num_heads",          type = int,        default = 6)
    parser.add_argument("--num_iterations",     type = int,        default = 2)
    parser.add_argument("--gpu",           	    type = int,        default = 0)
    parser.add_argument("--random_seed",        type = int,        default = 0)
    parser.add_argument("--model",              type = str,        default = "GCN_GSL_Cora")
    parser.add_argument("--description",        type = str,        default = "GCN GSL on Cora")
    return parser.parse_args()

def Graph_Construction(args, G, A_star):
    A = to_dense_adj(G.edge_index, max_num_nodes = args.num_nodes)[0]
    A_mask = 1 - (torch.matmul(G.train_mask.float().unsqueeze(1), G.train_mask.float().unsqueeze(0)) + torch.matmul((~G.train_mask).float().unsqueeze(1), (~G.train_mask).float().unsqueeze(0)))
    A_prime = A
    for k in range(args.num_iterations):
        A_dagger = (A_star - A_prime) * A_mask * 2708
        A_positive = F.gumbel_softmax(A_dagger, dim = 1, tau = 1, hard = True)
        A_positive[~G.train_mask, :] = 0
        A_positive = torch.logical_or(A_positive, A_positive.T).float()
        A_prime = A_prime + A_positive
    edge_index_prime, _ = dense_to_sparse(A_prime)
    print("Num Edges: {}".format(edge_index_prime.size(1)))
    G_prime = Data(x = G.x, edge_index = edge_index_prime, y = G.y, train_mask = G.train_mask, val_mask = G.val_mask, test_mask = G.test_mask)
    return G_prime

if __name__ == "__main__":

    args = set_args()
    seed_everything(args.random_seed)

    print("source_path: {}".format(args.source_path))
    print("target_path: {}".format(args.target_path))
    print("dataset: {}".format(args.dataset))
    print("num_epochs_0: {}".format(args.num_epochs_0))
    print("milestone_0: {}".format(args.milestone_0))
    print("milestone_1: {}".format(args.milestone_1))
    print("learning_rate_0: {}".format(args.learning_rate_0))
    print("num_heads: {}".format(args.num_heads))
    print("num_iterations: {}".format(args.num_iterations))
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

        dataset = Planetoid(root = os.path.join(args.source_path, args.dataset), name = "Cora")
        G = dataset[0].cuda(args.gpu)
        A = to_dense_adj(G.edge_index, max_num_nodes = G.x.size(0))[0]
        args.num_nodes = dataset[0].x.size(0)
        args.num_features = dataset[0].x.size(1)
        args.num_classes = 7
        print("Num Nodes: {}".format(args.num_nodes))
        print("Num Features: {}".format(args.num_features))
        print("Num Classes: {}".format(args.num_classes))
        print("Num Training Nodes: {}".format(G.train_mask.sum()))
        print("Num Validation Nodes: {}".format(G.val_mask.sum()))
        print("Num Testing Nodes: {}".format(G.test_mask.sum()))

        ########################################################## Training ##############################################################

        args.model_name = "0"
        model = Confusion_GSL(args, writer).cuda(args.gpu)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate_0, weight_decay = 5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.milestone_0, args.milestone_1] , gamma = 0.1)
        A_star = model.fit(G, criterion, optimizer, scheduler, args.num_epochs_0)
        print("Max Accuracy: {}".format(model.max_accuracy))

        ###################################################### Training - Prime ##########################################################

        for l in range(1, args.num_heads + 1):
            args.model_name = l
            model_prime = Confusion_GSL(args, writer).cuda(args.gpu)
            G_prime = Graph_Construction(args, G, A_star)
            G_prime.cuda(args.gpu)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model_prime.parameters(), lr = args.learning_rate_0, weight_decay = 5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.milestone_0, args.milestone_1] , gamma = 0.1)
            model_prime.fit(G_prime, criterion, optimizer, scheduler, args.num_epochs_1)
            print("Max Accuracy: {} | {}".format(model_prime.max_accuracy, l))

        ################################################## Experiment Recording #########################################################
		
        end_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        f.write("start_time: {}\n".format(start_time))
        f.write("end_time: {}\n".format(end_time))
        f.write("source_path: {}\n".format(args.source_path))
        f.write("target_path: {}\n".format(args.target_path))
        f.write("dataset: {}\n".format(args.dataset))
        f.write("num_epochs_0: {}\n".format(args.num_epochs_0))
        f.write("milestone_0: {}\n".format(args.milestone_0))
        f.write("milestone_1: {}\n".format(args.milestone_1))
        f.write("learning_rate_0: {}\n".format(args.learning_rate_0))
        f.write("num_heads: {}\n".format(args.num_heads))
        f.write("num_iterations: {}\n".format(args.num_iterations))
        f.write("gpu: {}\n".format(args.gpu))
        f.write("random_seed: {}\n".format(args.random_seed))
        f.write("model: {}\n".format(args.model))
        f.write("description: {}\n".format(args.description))
        f.write("max_accuracy: {}\n".format(model.max_accuracy))
        torch.save(model.state_dict(), os.path.join(args.result_path, "model.pth"))
