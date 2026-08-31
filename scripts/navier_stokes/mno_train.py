"""Train one recurrent neural operator and provide trajectory-loading helpers.

Run from ``scripts/navier_stokes``; for the paper configuration, for example::

    python3 mno_train.py --n_train 10000 --k_max 16 --n_layer 4 --df 64 --downsample 1 --n_roll_out 2

Raw ``../../data/navier_stokes/navier_stokes_*.npy`` trajectories and an
existing ``models/`` directory are required.  The active main block trains one
CLI-selected configuration for 500 epochs and writes the final checkpoint under
``models/MNO_model_*.pth`` (plus normalizer files only when normalization is
enabled).  ``preprocess_data`` and ``load_test_data`` are also intended for
import by ``mno_navier_stokes_solver.py``.
"""

import sys
import os
import math 
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

# Resolve project-local imports from the repository root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from nn.mno import setup_model, MNO_recurrent_train


def preprocess_data(n_train, n_test, nt, downsample, n_roll_out = 1):
    """
    Assemble autoregressive Navier–Stokes training and test samples.

    Each raw file has layout ``[forcing, omega_0, ..., omega_nt]``.  An input has
    channels ``[omega_t, forcing, x, y]`` and its target contains the next
    ``n_roll_out`` vorticity fields on the final axis.

    Parameters:
        n_train (int): Number of training samples to extract.
        n_test (int): Number of testing samples to extract.
        nt (int): Number of transitions considered in each trajectory.
        downsample (int): Exponent of the spatial stride, i.e. stride=2**downsample.
        n_roll_out (int): Number of consecutive future fields in each target.

    Returns:
        x_train, y_train: Training tensors with layouts ``[sample,nx,ny,4]``
            and ``[sample,nx,ny,n_roll_out]``.
        x_test, y_test: Test tensors with the same respective layouts.
        dx1, dx2 (float): Grid spacings in each direction (both equal).
    """
    # --- Constants derived from the raw data format ---
    n_file = 2000          # total number of available .npy files (0 … 1999)
    n = 256                # original spatial resolution (256×256 grid)
    
    
    # Every valid window must fit all n_roll_out targets within the trajectory.
    data_per_file = nt - n_roll_out + 1
    
    stride = 2**downsample 
    n = n // stride
    
    L = 1.0
    
    # Create coordinate grids (x1, x2) on the downsampled grid
    x1 = np.linspace(0.0, L, n, endpoint=False)
    x2 = np.linspace(0.0, L, n, endpoint=False)
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing="ij")
    dx1 = dx2 = L/n
    
    X, Y = [], []
    
    # Draw training trajectories from the beginning and held-out trajectories from
    # the end of the 2,000-file corpus; the slices below select the requested counts.
    for i in list(range(math.ceil(n_train / data_per_file))) + [n_file + x for x in range(-math.ceil(n_test / data_per_file), 0)]:
        data = np.load(f"../../data/navier_stokes/navier_stokes_{i:05d}.npy")
        # Raw layout: [forcing, omega_0, omega_1, ..., omega_nt, ...].
        vorticity = data[1:, ::stride, 0::stride]
        force     = data[0, ::stride, 0::stride]
        
        
        # Store static forcing/coordinates beside omega_t for each rollout window.
        for j in range(nt - n_roll_out + 1):
            X.append(np.stack([vorticity[j,:,:], force, x1_grid, x2_grid], axis=2))
            Y.append(vorticity[j+1:j+1+n_roll_out,:,:].transpose(1, 2, 0))
         
            
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))
    x_train, y_train  = X[:n_train,...], Y[:n_train,...] 
    x_test,  y_test   = X[-n_test:,...], Y[-n_test:,...] 

    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    print("dx1 = ", dx1, "dx2 = ", dx2)
    
    return x_train, y_train, x_test, y_test, dx1, dx2


def load_test_data(test_data_indices, nt, downsample):
    """
    Load complete trajectories for autoregressive evaluation.

    The returned channel order is ``[omega_t, forcing, x, y]`` at every saved time.

    Parameters:
        test_data_indices (list of int): testing file indices.
        nt (int): Number of transitions to load after the initial condition.
        downsample (int): Exponent of the spatial stride, i.e. stride=2**downsample.
        
    Returns:
        X (np.ndarray): Test data with layout [file, time, nx, ny, channel].
        dx1, dx2 (float): Grid spacings in each direction (both equal).
    """
    # --- Constants derived from the raw data format ---
    n_file = 2000          # total number of available .npy files (0 … 1999)
    n = 256                # original spatial resolution (256×256 grid)
    
    stride = 2**downsample 
    n = n // stride
    
    L = 1.0
    
    # Create coordinate grids (x1, x2) on the downsampled grid
    x1 = np.linspace(0.0, L, n, endpoint=False)
    x2 = np.linspace(0.0, L, n, endpoint=False)
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing="ij")
    dx1 = dx2 = L/n
    
    
    # Repeat the static forcing and coordinates along time so every state is a
    # self-contained four-channel model input.
    in_dim = 4
    x_test = []
    
    for i in test_data_indices:
        X = np.zeros((nt+1, n, n, in_dim))
        data_file_name = f"../../data/navier_stokes/navier_stokes_{i:05d}.npy"
        data = np.load(data_file_name)
        # Raw layout: [forcing, omega_0, omega_1, ..., omega_nt, ...].
        vorticity = data[1:, ::stride, 0::stride]
        force     = data[0, ::stride, 0::stride]
        
        X[:,...,0] = vorticity[:nt+1,...]
        X[:,...,1] = np.tile(force,   (nt+1, 1, 1))  
        X[:,...,2] = np.tile(x1_grid, (nt+1, 1, 1))    
        X[:,...,3] = np.tile(x2_grid, (nt+1, 1, 1))    
        x_test.append(X)      
    x_test = np.array(x_test)
    
    print("X.shape = ", x_test.shape)
    print("dx1 = ", dx1, "dx2 = ", dx2)
    
    return x_test, dx1, dx2



if __name__ == "__main__":


    # These CLI switches define the model capacity, spatial resolution, and
    # multi-step training horizon used in the checkpoint name below.

    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--df', type=int, default=64)
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--n_roll_out', type=int, default=2)

    args = parser.parse_args()

    k_max = args.k_max
    n_train = args.n_train
    n_layer = args.n_layer
    df = args.df
    downsample = args.downsample
    n_roll_out = args.n_roll_out

    save_model_name = f"models/MNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
    # Two-step rollout is the historical default and therefore has no suffix.
    if n_roll_out != 2:
        save_model_name += f"_nrollout{n_roll_out}"

    print("save_model_name = ", save_model_name )
    
    
    in_dim, out_dim = 4, 1
    n_test = 1000
    nt = 50
    x_train, y_train, x_test, y_test, dx1, dx2 = preprocess_data(n_train, n_test, nt, downsample, n_roll_out = n_roll_out)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim=in_dim, out_dim=out_dim, fc_dim=df, k_max=k_max, n_layer=n_layer, grad_layer=True, dxs=[dx1,dx2], dx_scale=10.0, pad_ratio=0.0, incremental=True)
    model = model.to(device)

    epochs = 500
    base_lr = 0.001
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = 8

    print('batch_size', batch_size, '\n')

    normalization_x = False
    normalization_y = False
    normalization_dim_x = [0,1] #channel-wise normalization
    normalization_dim_y = []    #normalization
    non_normalized_dim_x = 0
    non_normalized_dim_y = 0

    config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y,
                        "loss_p": 2}
                        }

    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MNO_recurrent_train(x_train, y_train, x_test, y_test, n_roll_out, config, model, save_model_name=save_model_name)
    

    
