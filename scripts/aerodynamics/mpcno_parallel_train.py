"""Launch distributed M-PCNO training for one surface resolution.

Run from ``scripts/aerodynamics`` with at least two CUDA devices::

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      mpcno_parallel_train.py --grad True --geo True --geointegral True \
      --n_layer 4 --k_max 16 --batch_size 4 --epochs 200 \
      --n_train 4000 --n_test 512 --dx_scale 10.0 --n_point 10000

The selected resolution must contain
``../../data/aerodynamics/PressureVTK_Processed_<N>/mpcno_data_n_train4000_n_test512.npz``.
The active ``__main__`` always calls ``main()``, which initializes DDP and saves
periodic/final model and output-normalizer checkpoints under ``models/``.
"""

import os
import torch
import sys
import argparse
from pathlib import Path
import gc

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from timeit import default_timer

from mpcno_helper import gen_data_tensors

sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from nn.geo_utility import compute_node_weight_scale
from nn.mpcno import compute_Fourier_modes, MPCNO, MPCNO_train_parallel

torch.set_printoptions(precision=16)




def setup_ddp(rank, local_rank, world_size):
    """Bind one process to one GPU and initialize the NCCL process group."""
    torch.cuda.set_device(local_rank)
    

    dist.init_process_group(
        backend="nccl",   
        init_method="env://",  
        rank=rank,
        world_size=world_size
    )
    
    # Offset seeds by global rank so stochastic work is not duplicated.
    torch.manual_seed(0 + rank)
    np.random.seed(0 + rank)
    
    if rank == 0:
        print(f"Rank {rank}: CUDA device {torch.cuda.current_device()}")    



def cleanup_ddp():
    """Release the distributed process group after training."""
    dist.destroy_process_group()



def train_ddp(rank, local_rank, world_size, args):
    """Load one resolution, construct the model, and run DDP training."""
    setup_ddp(rank, local_rank, world_size)
    
    # Parse configuration from parameters
    layer_selection = {'grad': args.grad.lower() == "true", 'geo': args.geo.lower() == "true", 'geointegral': args.geointegral.lower() == "true"}
    f_in_dim = 0
    f_out_dim = 1
    train_inv_L_scale = False
    k_max = args.k_max
    ndim = 3
    n_layer = args.n_layer
    layers = [64]*(n_layer+1)
    act = args.act
    dx_scale = args.dx_scale
    n_point = args.n_point
    n_train = args.n_train
    n_test  = args.n_test
    
    

    if rank == 0:
        print("Loading and preprocessing data...")
        
    # Every rank loads the full CPU archive; MPCNO_train_parallel constructs
    # DistributedSamplers that partition training and test samples by rank.
    save_model_name = f"models/MPCNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_npoint{n_point}"
    
    data_path = "../../data/aerodynamics/PressureVTK_Processed_"+str(n_point)
    data = np.load(data_path+"/mpcno_data_n_train"+str(n_train)+"_n_test"+str(n_test)+".npz")
    
    
    nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
    
    
    # Fourier-box side lengths enclose the approximate vehicle bounding box
    Ls = [10.0, 4.0, 3.2]
    
    node_weights = data["node_measures"]
    # Normalize surface quadrature weights by the two-dimensional measure scale
    # associated with the enclosing Fourier box.
    node_weight_scale = compute_node_weight_scale(2, Ls)
    node_weights = node_weights / node_weight_scale  
    
    node_weights = node_weights[...,0]
    
    # Match the gradient-branch scale used by the trained configuration.
    directed_edges, edge_gradient_weights = data["directed_edges"], data["edge_gradient_weights"] / dx_scale
    features = data["features"]

    
    ndata = nodes.shape[0]
    assert(ndata == n_train + n_test)
    
    # delete data and release its memory
    del data
    gc.collect()

    if rank == 0:
        print(nnodes.shape,node_mask.shape,nodes.shape,flush = True)
        print(args)
        print(f"ndata: {ndata},  n_train: {n_train}, n_test: {n_test}", flush=True)
        print("Casting to tensor", flush=True)
        
    nnodes = torch.from_numpy(nnodes)
    node_mask = torch.from_numpy(node_mask)
    nodes = torch.from_numpy(nodes.astype(np.float32))
    node_weights = torch.from_numpy(node_weights.astype(np.float32))
    features = torch.from_numpy(features.astype(np.float32))
    directed_edges = torch.from_numpy(directed_edges.astype(np.int64))
    edge_gradient_weights = torch.from_numpy(edge_gradient_weights.astype(np.float32))


    x_train, y_train, aux_train = gen_data_tensors(np.arange(n_train), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)
    x_test, y_test, aux_test = gen_data_tensors(np.arange(-n_test, 0), nodes, features, node_mask, node_weights, directed_edges, edge_gradient_weights, f_in_dim, f_out_dim)


    
    
    if rank == 0:
        print(f'x_train shape {x_train.shape}, x_test shape {x_test.shape}, y_train shape {y_train.shape}, y_test shape {y_train.shape}', flush = True)
        print('length of each dim: ',torch.amax(nodes, dim = [0,1]) - torch.amin(nodes, dim = [0,1]), flush = True)
        print(f'kmax = {k_max}')
        print(f'n_train = {n_train}, n_test = {n_test}')
        print(f'Ls = {Ls}')
        print(f'layer_selection = {layer_selection}')
        print(f'layers = {layers}')
        print(f'activation = {act}')


    modes = compute_Fourier_modes(ndim, [k_max, k_max, k_max], Ls)
    modes = torch.tensor(modes, dtype=torch.float, device=f'cuda:{local_rank}')
    model = MPCNO(ndim, modes,
                layer_selection = layer_selection,
                layers=layers,
                fc_dim=64,
                in_dim=x_train.shape[-1], out_dim=y_train.shape[-1],
                act = act,
                ).to(local_rank)

    # DDP synchronizes parameter gradients; data partitioning occurs in the
    # training routine through one DistributedSampler per split.
    ddp_model = DDP(model, device_ids=[local_rank])

    epochs = args.epochs
    base_lr = 5e-4
    lr_ratio = 10
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = args.batch_size
    if rank == 0:
        print(f'batch_size = {batch_size}')

    # Coordinates and normals remain in physical units; only Cp is standardized.
    normalization_x = False
    normalization_y = True
    normalization_dim_x = []
    normalization_dim_y = []
    non_normalized_dim_x = 4
    non_normalized_dim_y = 0


    config = {"train" : {"base_lr": base_lr, 'lr_ratio': lr_ratio, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y,
                        "loss_p": 1.0}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MPCNO_train_parallel(
        x_train, aux_train, y_train, x_test, aux_test, y_test, config, ddp_model, rank=rank, local_rank = local_rank, world_size=world_size, save_model_name=save_model_name)
    
    
    cleanup_ddp()




    
    
def main():
    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--grad', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--geo', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--geointegral', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    # Preprocess data n_each from each subcategories
    parser.add_argument('--n_train', type=int, default=900)
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--act', type=str, default="gelu")
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dx_scale', type=float, default=10.0)
    parser.add_argument('--n_point', type=int, default=10000)
    
    args = parser.parse_args()
    

    # torchrun supplies global rank, node-local rank, and world size.
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    train_ddp(rank, local_rank, world_size, args)
    
if __name__ == "__main__":
    main()
