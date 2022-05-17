import os
import torch
import pandas as pd
from sklearn.datasets import load_iris
from torch_geometric.data import Dataset, InMemoryDataset

class UCI_Dataset(InMemoryDataset):
	def __init__(self, root, transform = None, pre_transform = None, pre_filter = None):
		super().__init__(root, transform, pre_transform, pre_filter)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return ["data.csv"]

	@property
	def processed_file_names(self):
		return ["data.pt"]

	@property
	def num_classes(self):
		y = self.data.y
		if y is None:
			return 0
		else:
			return torch.unique(y).numel()

	@property
	def num_node_features(self):
		x = self.data.x
		if x is None:
			return 0
		else:
			return x.size(1)
		
	def download(self):
		data = load_iris()
		df_X = pd.DataFrame(data["data"], columns = data["feature_names"])
		df_y = pd.DataFrame(data["target"], columns = ["target"])
		df = pd.concat([df_X, df_y], axis = 1)
		df.to_csv(os.path.join(self.raw_dir, self.raw_paths[0]), index = False)

	def process(self):
		df = pd.read_csv(os.path.join(self.raw_dir, self.raw_paths[0]))
		if self.pre_transform is not None:
			data_list = [self.pre_transform(row) for idx, row in df.iterrows()]
		data, slices = self.collate(data_list)
		torch.save((data, slices), os.path.join(self.processed_dir, self.processed_paths[0]))
