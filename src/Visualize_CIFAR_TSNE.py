import os
import sys
import torch
import random
import argparse
from tqdm import tqdm as tqdm
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything

from lib.util_visualize import *


def set_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_path",        type = str,        default = "../log/Confusion_GSL_CIFAR/2022_05_23_15_08_51/")
	parser.add_argument("--source_file",        type = str,        default = "")
	parser.add_argument("--target_path",        type = str,        default = "../log/Confusion_GSL_CIFAR/2022_05_23_15_08_51/")
	parser.add_argument("--target_file",        type = str,        default = "")
	parser.add_argument("--num_classes",        type = int,        default = 10)
	parser.add_argument("--gpu",                type = int,        default = 0)
	parser.add_argument("--random_seed",        type = int,        default = 0)
	return parser.parse_args()


if __name__ == "__main__":

    args = set_args()
    args.result_path = args.target_path
			
    seed_everything(args.random_seed)

    print("source_path: {}".format(args.source_path))
    print("target_path: {}".format(args.target_path))
    print("random_seed: {}".format(args.random_seed))

    for fname in tqdm(os.listdir(args.source_path)):
        if ("Graph" in fname) and ("pth" in fname):
            data = torch.load(os.path.join(args.source_path, fname))
            visualize_TSNE(args, data, fname.split(".")[0] + ".png")
