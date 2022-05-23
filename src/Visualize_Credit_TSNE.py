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
	parser.add_argument("--source_path",        type = str,        default = "../log/Confusion_GSL_Credit/2022_05_23_19_19_19/")
	parser.add_argument("--source_file",        type = str,        default = "")
	parser.add_argument("--target_path",        type = str,        default = "../log/Confusion_GSL_Credit/2022_05_23_19_19_19/")
	parser.add_argument("--target_file",        type = str,        default = "")
	parser.add_argument("--num_classes",        type = int,        default = 2)
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
        if ("features" in fname):
            try:
                print(x.shape)
                x = torch.cat([x, torch.load(os.path.join(args.source_path, fname))], dim = 1)
            except NameError:
                x = torch.load(os.path.join(args.source_path, fname))
        elif ("labels" in fname):
            y = torch.load(os.path.join(args.source_path, fname))
    visualize_TSNE_(args, x, y, "Credit_TSNE.png")
