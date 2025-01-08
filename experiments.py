from argparse import Namespace
import torch
import sys
import tqdm
from typing import Tuple
from torch import Tensor
from torch_geometric.transforms import AddRandomWalkPE

from helpers.classes import ModelArgs
from model import Model
from helpers.utils import set_seed


class Experiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=self.seed)

        # modifications for the tiger dataset
        self.multi_graph_samples = self.num_graph_samples > 1

    def run(self):
        # load model
        model_args = ModelArgs(model_type=self.model_type, num_layers=self.num_layers,
                               in_dim=self.in_dim + self.rw_pos_length,
                               hidden_dim=self.hidden_dim, out_dim=self.out_dim, pool=self.pool)
        model = Model(model_args=model_args).to(device=self.device)

        # load datasets
        # (out_dim,), (out_dim,), (,)
        mean_of_scores, std_of_scores, std_distance = self.multi_sample_test(model=model)  # (out_dim,)
        print(f'Final mean_of_scores={mean_of_scores}')
        print(f'Final std_of_scores={std_of_scores}')
        print(f'Final std_distance={std_distance}')

    def set_neighbourhood(self, data, neighbourhood_type):
      num_nodes_by_neighbourhood_type = {'isolated' : 1, 'edge' : 2, 'two_edges' : 3}
      num_fixed_nodes = num_nodes_by_neighbourhood_type[neighbourhood_type]

      edge_index = torch.transpose(data.edge_index, 0, 1).tolist()
      new_edge_index = []
      for v, u in edge_index:
        if v >= num_fixed_nodes and u >= num_fixed_nodes:
          new_edge_index.append([v, u])

      if neighbourhood_type == 'edge':
        new_edge_index.append([0, 1])
        new_edge_index.append([1, 0])
      elif neighbourhood_type == 'two_edges':
        new_edge_index.append([0, 1])
        new_edge_index.append([1, 0])
        new_edge_index.append([1, 2])
        new_edge_index.append([2, 1])
            
      if len(new_edge_index) == 0:
        data.edge_index = torch.empty(2, 0, dtype=data.edge_index.dtype)
      else:
        data.edge_index = torch.transpose(torch.tensor(new_edge_index), 0, 1)


    def multi_sample_test(self, model: Model) -> Tuple[Tensor, Tensor, Tensor]:
        # load model
        model.eval()
        pos_enc_transform = AddRandomWalkPE(walk_length=self.rw_pos_length) if self.rw_pos_length > 0 else None

        prefix_str = "Results/Temp"
        scores = torch.empty(size=(self.out_dim, 0))
        node_embeddings = None

        for _ in range(self.seed_for_fixed_nodes + 1):
          global_x = torch.rand(size=(5, self.in_dim,))

        with tqdm.tqdm(total=self.num_graph_samples, file=sys.stdout) as pbar:
            for sample_idx in range(self.num_graph_samples):
                data = self.dataset.load(num_nodes=self.graph_size, in_dim=self.in_dim,
                                         seed=self.seed + sample_idx, pos_enc_transform=pos_enc_transform)
                
                if self.fix_neighbourhood is not None:
                  self.set_neighbourhood(data, self.fix_neighbourhood)

                if self.fix_input_features > 0:
                  data.x[:self.fix_input_features,:] = global_x[:self.fix_input_features,:]

                score = model(data.x.to(device=self.device),
                              edge_index=data.edge_index.to(device=self.device)).detach().cpu()  # (out_dim,)
                node_predictions_for_sample = model.get_node_predictions(data.x.to(device=self.device),
                              edge_index=data.edge_index.to(device=self.device)).detach().cpu() # (num_nodes, out_dim)
                scores = torch.cat((scores, score.unsqueeze(dim=1)), dim=1)  # (out_dim, num_graph_samples)
                node_predictions = node_predictions_for_sample.unsqueeze(dim=2) if sample_idx == 0 else torch.cat((node_predictions, node_predictions_for_sample.unsqueeze(dim=2)), dim=2)  # (num_nodes, out_dim, num_graph_samples)

                # prints
                pbar.set_description(f'sample: {sample_idx}/{self.num_graph_samples}')
                pbar.update(n=1)
        
        mean_per_dim = torch.mean(scores, dim=1)  # (out_dim,)
        distance_from_mean = torch.norm(scores - mean_per_dim.unsqueeze(dim=1), dim=0, p=2)  # (num_graph_samples,)
        std_of_distance_from_mean = torch.std(distance_from_mean, dim=0)  # (,)

        cnt=5
        print('predictions_means=[[', end='')
        for i in range(cnt):
          print(torch.mean(node_predictions[i,:,:], dim=1).numpy().tolist(), end=(', ' if i < cnt-1 else ''))
        print(']]')

        print('predictions_stds=[[', end='')
        for i in range(cnt):
          print(torch.std(node_predictions[i,:,:], dim=1).numpy().tolist(), end=(', ' if i < cnt-1 else ''))
        print(']]')

        return torch.mean(scores, dim=1), torch.std(scores, dim=1), std_of_distance_from_mean
