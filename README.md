# Convergence of Graph Neural Networks on Random Graphs in Node-Level Tasks

This repository contains the source code for the Graph Representation Learning course mini-project.

The codebase is adapted from the repository of **[Almost Surely Asymptotically Constant Graph Neural Networks](https://arxiv.org/abs/2403.03880)** (https://github.com/benfinkelshtein/GNN-Asymptotically-Constant).

The experiment code is available at [Experiments.ipynb](https://github.com/AlesyaIvanova/GNN-Asymptotically-Constant/blob/main/Experiments.ipynb).

## Installation ##
To reproduce the results please use Python 3.9, PyTorch version 2.0.0, Cuda 11.8, PyG version 2.3.0, and torchmetrics.

```bash
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric==2.3.0
pip install ogb
```

## Datasets

### datasets

ER, LogER and InverseER represent the $`ER(n, p(n) = 0.1)`$, $`ER(n, p(n) = \frac{\log{n}}{n})`$
and $`ER(n, p(n) = \frac{1}{50n})`$ distributions.

## Running

The script we use to run the experiments is ``./main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

The parameters of the script are:

- ``--dataset``: name of the dataset.
The available options are: ER, LogER, InverseER, SBM, BA, Tiger1k, Tiger5k, Tiger10k, Tiger25k and Tiger90k.

- ``--graph_size``: the graph size.
- ``--num_graph_samples``: the number of different graph size samples taken. 
- ``--rw_pos_length``: the maximal length of the random walk in the Random Walk Positional Encoding.
- ``--model_type``: the type of model that is used.
The available options are: MEAN_GNN, GCN, GAT and GPS.

- ``--num_layers``: the network's number of layers.
- ``--in_dim``: the network's input dimension.
- ``--hidden_dim``: the network's hidden dimension.
- ``--output_dim``: the network's output dimension.
- ``--pool``: name of the graph pooling.

- ``--seed``: a seed to set random processes.
- ``--gpu``: the number of the gpu that is used to run the code on.

- ``--fix_input_features``: the number of nodes with fixed features (new).
- ``--fix_neighbourhood``: the type of fixed neighbourhood (new).
- ``--seed_for_fixed_nodes``: a seed for the fixed nodes' features initialization (new).
  
## Example running

To perform experiments on the LogER dataset using a MEAN_GNN with 3 layers, an output dimension of 5, and input and hidden dimensions of 128, with the fixed neighbourhood of the first node consisting of 3 nodes connected by 2 edges. See an example usage of the following command:
 
```bash
python -u main.py --dataset LogER --model_type MEAN_GNN --in_dim 128 --hidden_dim 128 --out_dim 5 --num_layers 3 --seed 0 --fix_input_features 3 --fix_neighbourhood 'two_edges' --seed_for_fixed_nodes 1
```
