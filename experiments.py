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


    def multi_sample_test(self, model: Model) -> Tuple[Tensor, Tensor, Tensor]:
        # load model
        model.eval()
        pos_enc_transform = AddRandomWalkPE(walk_length=self.rw_pos_length) if self.rw_pos_length > 0 else None

        prefix_str = "Results/Temp"
        scores = torch.empty(size=(self.out_dim, 0))
        node_embeddings = None

        # global_x = torch.rand(size=(self.in_dim,))
        # print(global_x)

        with tqdm.tqdm(total=self.num_graph_samples, file=sys.stdout) as pbar:
            for sample_idx in range(self.num_graph_samples):
                data = self.dataset.load(num_nodes=self.graph_size, in_dim=self.in_dim,
                                         seed=self.seed + sample_idx, pos_enc_transform=pos_enc_transform)
                
                # if sample_idx < 5:
                #   print(data.x[0,:])
                # if self.same_node_features_for_all_graph_samples:
                #   data.x[0,:] = global_x
                #   if global_x == None:
                #     global_x = data.x.clone()
                #   else:
                #     data.x = global_x.clone()

                score = model(data.x.to(device=self.device),
                              edge_index=data.edge_index.to(device=self.device)).detach().cpu()  # (out_dim,)
                node_predictions_for_sample = model.get_node_predictions(data.x.to(device=self.device),
                              edge_index=data.edge_index.to(device=self.device)).detach().cpu() # (num_nodes, out_dim)
                scores = torch.cat((scores, score.unsqueeze(dim=1)), dim=1)  # (out_dim, num_graph_samples)
                node_predictions = node_predictions_for_sample.unsqueeze(dim=2) if sample_idx == 0 else torch.cat((node_predictions, node_predictions_for_sample.unsqueeze(dim=2)), dim=2)  # (num_nodes, out_dim, num_graph_samples)

                # prints
                pbar.set_description(f'sample: {sample_idx}/{self.num_graph_samples}')
                pbar.update(n=1)
        # (out_dim,), (out_dim,), (num_graph_samples,)
        # scores = node_predictions[0,:,:]
        # print(node_predictions.shape)
        mean_per_dim = torch.mean(scores, dim=1)  # (out_dim,)
        distance_from_mean = torch.norm(scores - mean_per_dim.unsqueeze(dim=1), dim=0, p=2)  # (num_graph_samples,)
        std_of_distance_from_mean = torch.std(distance_from_mean, dim=0)  # (,)

        # node_embeddings_for_node0 = node_embeddings[0,:,:]
        # node_embeddings_mean_per_dim = torch.mean(node_embeddings_for_node0, dim=1)

        for i in range(5):
          print(torch.mean(node_predictions[i,:,:]).item(), torch.std(node_predictions[i,:,:]).item())

        return torch.mean(scores, dim=1), torch.std(scores, dim=1), std_of_distance_from_mean
