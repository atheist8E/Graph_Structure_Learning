import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from tsnecuda import TSNE
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def visualize_TSNE(args, G, fname):
    x = G.x.detach().cpu().numpy()
    y = G.y.detach().cpu().numpy()
    x_embedded = TSNE(n_components = 2, perplexity = 5, random_seed = args.random_seed).fit_transform(x)
    df_subset = pd.DataFrame(x_embedded, columns = ["x1_embedded", "x2_embedded"])
    df_subset["y"] = y
    sns.scatterplot(
        x = "x1_embedded", y = "x2_embedded",
        hue = "y",
        palette = sns.color_palette("bright", args.num_classes),
        data = df_subset,
        legend = "full",
    )
    plt.savefig(os.path.join(args.result_path, fname))
    plt.close()

def visualize_TSNE_(args, x, y, fname):
    x = x.numpy()
    y = y.numpy()
    x_embedded = TSNE(n_components = 2, perplexity = 400, random_seed = args.random_seed).fit_transform(x)
    df_subset = pd.DataFrame(x_embedded, columns = ["x1_embedded", "x2_embedded"])
    df_subset["y"] = y
    sns.scatterplot(
        x = "x1_embedded", y = "x2_embedded",
        hue = "y",
        palette = sns.color_palette("bright", args.num_classes),
        data = df_subset,
        legend = "full",
    )
    plt.savefig(os.path.join(args.result_path, fname))
    plt.close()
