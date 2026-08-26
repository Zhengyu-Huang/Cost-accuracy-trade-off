"""Train one modified neural operator (MNO) for the Darcy benchmark.

Run from ``scripts/darcy``. Training reads the requested leading samples and
the final 1,000 test samples from ``../../data/darcy`` and writes the model and
normalizer checkpoints below ``models/``. A CUDA device is used when available.
Example: ``python3 mno_train.py --n_train 4000 --k_max 16 --n_layer 4 --df 64 --downsample 2``.
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

# Add the project root so the shared neural-operator modules can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from nn.mno import MNO2d, MNO_train, setup_model


np.set_printoptions(precision=10, suppress=True)

def preprocess_data(n_train, n_test, downsample = 1):
    """Load disjoint training/test sets and construct operator input channels.

    Each input has shape ``(nx + 1, ny + 1, 3)`` with permeability and two
    coordinate channels; each target has one pressure-solution channel.
    ``downsample`` is the exponent in the spatial stride ``2**downsample``.
    """
    n_file = 10000
    
    # Raw fields have 512 cells and therefore 513 nodes per direction.
    n = 512
    stride = 2**downsample 
    
    n = n // stride
    
    L = 1.0
    
    x1 = np.linspace(0.0, L, n+1, endpoint=True)
    x2 = np.linspace(0.0, L, n+1, endpoint=True)
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing="ij")
    dx1 = dx2 = L/n
    
    X, Y = [], []
    # Train on the first n_train files and test on the final n_test files.
    for i in list(range(n_train)) + [n_file + x for x in range(-n_test, 0)]:
        data = np.load(f"../../data/darcy/darcy_data_{i:05d}.npy")
        # The final axis stores permeability and the reference solution.
        data = data[0::stride, 0::stride, :]
        X.append(np.stack([data[:,:,0], x1_grid, x2_grid], axis=2))    # kappa, x, y
        Y.append(data[:,:,1:])                                         # u
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))
    x_train, y_train  = X[:n_train,...], Y[:n_train,...] 
    x_test,  y_test   = X[-n_test:,...], Y[-n_test:,...] 

    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    print("dx1 = ", dx1, "dx2 = ", dx2)
    
    return x_train, y_train, x_test, y_test, dx1, dx2


def load_test_data(test_data_indices, downsample):
    """
    Preprocess Darcy flow simulation data for testing.

    The raw data are stored in multiple .npy files, each containing the
    permeability field and the solution. This function loads
    the required files, optionally downsamples the spatial grid

    Parameters:
        test_data_indices (list of int): testing file indices.
        downsample (int): Exponent in the spatial stride ``2**downsample``.
        
    Returns:
        x_test (np.ndarray): Inputs shaped [n_test, nx + 1, ny + 1, 3].
        y_test (np.ndarray): Targets shaped [n_test, nx + 1, ny + 1, 1].
        dx1, dx2 (float): Grid spacings in each direction (both equal).
    """
    # --- Constants derived from the raw data format ---
    n_file = 10000          # total number of available .npy files
    assert all(0 <= idx < 10000 for idx in test_data_indices)
    n = 512                 # original number of cells per direction
    
    stride = 2**downsample 
    n = n // stride
    
    L = 1.0
    
    # Create coordinate grids (x1, x2) on the downsampled grid
    x1 = np.linspace(0.0, L, n+1, endpoint=True)
    x2 = np.linspace(0.0, L, n+1, endpoint=True)
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing="ij")
    dx1 = dx2 = L/n
    
    
    X, Y = [], []
    for i in test_data_indices:
        data = np.load(f"../../data/darcy/darcy_data_{i:05d}.npy")
        data = data[0::stride, 0::stride, :]
        # The final axis stores permeability and the reference solution.
        
        X.append(np.stack([data[:,:,0], x1_grid, x2_grid], axis=2))    # kappa, x, y
        Y.append(data[:,:,1:])     
        
    x_test,  y_test   = np.array(X), np.array(Y)

    
    print("x_test.shape = ", x_test.shape, "y_test.shape = ", y_test.shape)
    print("dx1 = ", dx1, "dx2 = ", dx2)
    
    return x_test, y_test, dx1, dx2





if __name__ == "__main__":
    
    
    # Parse one training configuration; shell scripts can sweep these options.

    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--df', type=int, default=64)
    parser.add_argument('--downsample', type=int, default=2)
    args = parser.parse_args()

    k_max = args.k_max
    n_train = args.n_train
    n_layer = args.n_layer
    df = args.df
    downsample = args.downsample

    save_model_name = f"models/MNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
    print("save_model_name = ", save_model_name )
    
    in_dim, out_dim = 3, 1 
    n_test = 1000
    x_train, y_train, x_test, y_test, dx1, dx2 = preprocess_data(n_train, n_test, downsample=downsample)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim=in_dim, out_dim=out_dim, fc_dim=df, k_max=k_max, n_layer=n_layer, grad_layer=True, dxs=[dx1,dx2], dx_scale=10.0, pad_ratio=0.1, incremental=False)
    model = model.to(device)
    
    epochs = 500
    base_lr = 0.001
    scheduler = "OneCycleLR"
    weight_decay = 1.0e-4
    batch_size = 8

    print('batch_size', batch_size, '\n')

    normalization_x = True
    normalization_y = True
    normalization_dim_x = [0,1] # channel-wise normalization
    normalization_dim_y = []
    non_normalized_dim_x = 0
    non_normalized_dim_y = 0

    config = {"train" : {"base_lr": base_lr, "weight_decay": weight_decay, "epochs": epochs, "scheduler": scheduler,  "batch_size": batch_size, 
                        "normalization_x": normalization_x,"normalization_y": normalization_y, 
                        "normalization_dim_x": normalization_dim_x, "normalization_dim_y": normalization_dim_y, 
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y,
                        "loss_p": 2}
                        }


    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MNO_train(x_train, y_train, x_test, y_test, config, model, save_model_name=save_model_name)
    

    

