import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import BaseTransform

class Isolate(BaseTransform):
	def __init__(self):
		pass

	def __call__(self, graph):
		loop_index = torch.arange(0, graph.num_nodes, dtype = torch.long, device = graph.x.device)
		loop_index = loop_index.unsqueeze(0).repeat(2, 1)
		graph.edge_index = loop_index
		return graph

class KNN(BaseTransform):
	def __init__(self, k):
		self.k = k

	def __call__(self, graph):
		graph.edge_attr = None
		graph.edge_index = knn_graph(graph.x, self.k, loop = True)
		return graph

class Tabular_to_Graph(BaseTransform):
	def __init__(self):
		pass

	def __call__(self, row):
		X = row.iloc[:-1]
		y = row.iloc[-1]
		loop_index = torch.arange(0, len(X), dtype = torch.long)
		loop_index = loop_index.unsqueeze(0).repeat(2, 1)
		X =	torch.tensor(X, dtype = torch.float32).unsqueeze(1)
		y = torch.tensor(y, dtype = torch.long)
		graph = Data(x = X, edge_index = loop_index, y = y)
		return graph
