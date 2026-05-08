import sys
import os
import math 
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from nn.mno import setup_model, MNO_recurrent_train


def preprocess_data(n_train, n_test, nt, downsample, n_roll_out = 1):
    """
    Preprocess Navier‑Stokes simulation data for training and testing.

    The raw data are stored in multiple .npy files, each containing a sequence of
    vorticity fields over time plus a constant forcing field. This function loads
    the required files, optionally downsamples the spatial grid, and assembles
    input/output pairs for autoregressive and reccusive training 
    x(0) -> x(1) -> x(2) ... ->  x(n_roll_out) 
    

    Parameters:
        n_train (int): Number of training samples to extract.
        n_test (int): Number of testing samples to extract.
        nt = 20 (int): Number of time steps load from per file (excluding initial condition)
        downsample (int): Downsampling factor (2**downsample). Default 1 => no downsampling.
        roll_out (bool): If True, create consecutive time‑step pairs for rollout
                         training; if False, use only the first and the nt-th time steps.

    Returns:
        x_train (torch.Tensor [n_train, ...., in_dim])
        y_train (torch.Tensor [n_test, ...., out_dim * n_roll_out]): Training input/output tensors.
        x_test (torch.Tensor [n_test, ...., in_dim])
        y_test (torch.Tensor [n_test, ...., out_dim * n_roll_out]): Test input/output tensors.
        dx1, dx2 (float): Grid spacings in each direction (both equal).
    """
    # --- Constants derived from the raw data format ---
    n_file = 2000          # total number of available .npy files (0 … 1999)
    n = 256                # original spatial resolution (256×256 grid)
    
    
    # Number of data samples contributed by one file
    # If roll_out: each file provides 'nt' consecutive pairs (20 samples)
    # If not rolling: each file provides only 1 pair (first and last time step)
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
    
    # --- Determine which file indices to load ---
    # Training files: first ceil(n_train / data_per_file) files (starting from index 0)
    # Test files: last ceil(n_test / data_per_file) files (negative indices, e.g., -1, -2, ...)
    for i in list(range(math.ceil(n_train / data_per_file))) + [n_file + x for x in range(-math.ceil(n_test / data_per_file), 0)]:
        data = np.load(f"../../data/navier_stokes/navier_stokes_{i:05d}.npy")
        # data : nt+2 by n by n array. 
        # forcing, vorticity_0, vorticity_1, ......, vorticity_nt
        vorticity = data[1:, ::stride, 0::stride]
        force     = data[0, ::stride, 0::stride]
        
        
        # Create nt consecutive pairs (vorticity[t], vorticity[t+1])
        for j in range(nt - n_roll_out + 1):
            X.append(np.stack([vorticity[j,:,:], force, x1_grid, x2_grid], axis=2))           # omega, force, x, y
            Y.append(vorticity[j+1:j+1+n_roll_out,:,:].transpose(1, 2, 0))                    # omega'
         
            
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Y.astype(np.float32))
    x_train, y_train  = X[:n_train,...], Y[:n_train,...] 
    x_test,  y_test   = X[-n_test:,...], Y[-n_test:,...] 

    print("x_train.shape = ", x_train.shape, "y_train.shape = ", y_train.shape)
    print("dx1 = ", dx1, "dx2 = ", dx2)
    
    return x_train, y_train, x_test, y_test, dx1, dx2


def load_test_data(n_test_file, nt, downsample):
    """
    Preprocess Navier‑Stokes simulation data for testing.

    The raw data are stored in multiple .npy files, each containing a sequence of
    vorticity fields over time plus a constant forcing field. This function loads
    the required files, optionally downsamples the spatial grid, for roll-out test

    Parameters:
        n_test_file (int): Number of testing filess to extract.
        downsample (int): Downsampling factor (2**downsample). Default 1 => no downsampling.
        
    Returns:
        X (torch.Tensor): Test tensors [n_test_file, nt+1, nx, ny, in_dim].
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
    
    
    # --- Determine which file indices to load ---
    # Training files: first ceil(n_train / data_per_file) files (starting from index 0)
    # Test files: last ceil(n_test / data_per_file) files (negative indices, e.g., -1, -2, ...)
    in_dim = 4
    x_test = []
    
    for i in [n_file + x for x in range(-n_test_file, 0)]:
        X = np.zeros((nt+1, n, n, in_dim))
        data_file_name = f"../../data/navier_stokes/navier_stokes_{i:05d}.npy"
        data = np.load(data_file_name)
        # data : nt+2 by n by n array. 
        # forcing, vorticity_0, vorticity_1, ......, vorticity_nt
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


    ###################################
    # load parameters
    ###################################

    parser = argparse.ArgumentParser(description='Train model with different configurations and options.')

    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--df', type=int, default=64)
    parser.add_argument('--downsample', type=int, default=1)
    args = parser.parse_args()

    k_max = args.k_max
    n_train = args.n_train
    n_layer = args.n_layer
    df = args.df
    downsample = args.downsample

    save_model_name = f"models/MNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
    print("save_model_name = ", save_model_name )
    
    
    in_dim, out_dim = 4, 1
    n_test = 1000
    nt = 50
    n_roll_out = 2
    x_train, y_train, x_test, y_test, dx1, dx2 = preprocess_data(n_train, n_test, nt, downsample, n_roll_out = n_roll_out)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = setup_model(in_dim=in_dim, out_dim=out_dim, fc_dim=df, k_max=k_max, n_layer=n_layer, dxs=[dx1,dx2], dx_scale=10.0, pad_ratio=0.0, incremental=True)
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
                        "non_normalized_dim_x": non_normalized_dim_x, "non_normalized_dim_y": non_normalized_dim_y}
                        }

    train_rel_l2_losses, test_rel_l2_losses, test_l2_losses = MNO_recurrent_train(x_train, y_train, x_test, y_test, n_roll_out, config, model, save_model_name=save_model_name)
    

    



