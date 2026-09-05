"""Benchmark trained FNO checkpoints on Darcy-flow test samples.

Run from ``scripts/darcy_fno``. The default driver requires CUDA, the final
100 raw files in ``../../data/darcy``, and all nine matching checkpoints plus
normalizer states in ``models/``. It writes
``data/cost_accuracy_fno_solver_data.npz`` after CPU and GPU evaluation.
Example: ``python3 fno_darcy_solver.py``.
"""

import sys
import os
import math 
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer
from fno_train import load_test_data

# Add the project root so the shared model and utility modules can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from nn.mno import setup_model, mno_floating_point_cost 
from utility.normalizer import UnitGaussianNormalizer


np.set_printoptions(precision=10, suppress=True)




def cost_accuracy_fno_solver_helper(device, downsample, k_max_values, n_layer_values, df_values, n_train, n_trial):
    """Benchmark FNO configurations on held-out Darcy-flow samples.

    The returned cost tensor is indexed by spectral cutoff, layer count,
    feature width, trial, and ``[FLOP estimate, device timer]``. Accuracy has
    the same configuration/trial axes and stores relative L2 error.
    """
    
    nx = ny = 512
    nx, ny = nx//(2**downsample), ny//(2**downsample)
    
    ngrid = nx + 1
    L = 1.0
    cost, accuracy = np.zeros((len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2)), np.zeros((len(k_max_values), len(n_layer_values), len(df_values), n_trial))
    
    dim, in_dim, out_dim = 2, 3, 1 
    x_test, y_test, dx1, dx2 = load_test_data(np.arange(10000-n_trial, 10000), downsample=downsample)

    
    normalization_x = True
    normalization_y = True
    

    ne = ngrid * ngrid
    
    n_repeat = 10
    for k_max_index, k_max in enumerate(k_max_values):
        for n_layer_index, n_layer in enumerate(n_layer_values):
            for df_index, df in enumerate(df_values):
                
                checkpoint_path = f"models/FNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
                
                model = setup_model(in_dim=in_dim, out_dim=out_dim, fc_dim = df, k_max=k_max, n_layer=n_layer, grad_layer=False, dxs=[dx1,dx2], dx_scale=10.0, pad_ratio=0.1, checkpoint_path=checkpoint_path+".pth")
                model = model.to(device)

                if normalization_x:
                    x_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(checkpoint_path + "_normalization_x.pth", map_location="cpu", weights_only=True,), device=device)
                if normalization_y:
                     y_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(checkpoint_path + "_normalization_y.pth", map_location="cpu", weights_only=True,), device=device)
     
                
                for i in range(n_trial):
                    # Preserve the batch axis: (1, nx + 1, ny + 1, 3).
                    xi = torch.from_numpy(x_test[[i],...].astype(np.float32)).to(device) 
                    # Run one untimed inference to initialize kernels and caches.
                    x = x_normalizer.encode(xi)
                    y_pred = model(x)
                    if normalization_y:
                        y_pred = y_normalizer.decode(y_pred)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start_time = time.perf_counter()
                    for j in range(n_repeat):
                        x = x_normalizer.encode(xi)
                        y_pred = model(x)
                        if normalization_y:
                            y_pred = y_normalizer.decode(y_pred)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    end_time = time.perf_counter()
                    solve_time = (end_time - start_time)/n_repeat
    
                    y_ref = y_test[[i],...]
                    
                    
                    y_pred, y_ref = y_pred.detach().cpu().numpy(), y_ref
                    rel_error = np.linalg.norm(y_pred - y_ref)/np.linalg.norm(y_ref)
                    # FLOPs are estimated analytically from the model/grid dimensions.
                    cost[k_max_index, n_layer_index, df_index, i, 0] = mno_floating_point_cost(dim, in_dim, out_dim, k_max, df, n_layer, ne, grad_layer=False)
                    cost[k_max_index, n_layer_index, df_index, i, 1] = solve_time
                    accuracy[k_max_index, n_layer_index, df_index, i] = rel_error
                    
                    print("relative error is : ", rel_error, " flops = ", cost[k_max_index, n_layer_index, df_index, i, 0], " computing_time = ", solve_time)

                
    
    return  cost, accuracy 


def cost_accuracy_fno_solver(downsample_values, k_max_values, n_layer_values, df_values, n_train, n_trial = 10):
    """Benchmark each grid/model configuration on both CUDA and CPU.

    The final cost axis is ``[FLOP estimate, CPU seconds, GPU seconds]``.
    """
    
    cost = np.zeros((len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 3)) 
    accuracy = np.zeros((len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial))
    
    for downsample_index, downsample in enumerate(downsample_values):
        # Evaluate identical samples/configurations on each device for timing.
        for device in [torch.device('cuda') , torch.device('cpu')]:
            cost_ds, accuracy_ds  = cost_accuracy_fno_solver_helper(device, downsample, k_max_values = k_max_values, n_layer_values = n_layer_values, df_values = df_values, n_train = n_train, n_trial = n_trial)
            cost[downsample_index, :, :, :, :, 0] = cost_ds[...,0]
            if device.type == 'cpu':
                cost[downsample_index, :, :, :,:, 1] = cost_ds[...,1]
            else:
                cost[downsample_index, :, :, :,:, 2] = cost_ds[...,1]
                
            accuracy[downsample_index, ...] = accuracy_ds
             
                

    np.savez_compressed('data/cost_accuracy_fno_solver_data.npz', cost=cost, accuracy=accuracy)

    return  cost, accuracy 



def fno_solver(test_index, downsample):
    """Run the fixed trained FNO configuration and save one prediction."""
    
    nx = ny = 512
    nx, ny = nx//(2**downsample), ny//(2**downsample)
    
    ngrid = nx + 1
    L = 1.0
    
    dim, in_dim, out_dim = 2, 3, 1 
    x_test, y_test, dx1, dx2 = load_test_data([test_index], downsample=downsample)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    normalization_x = True
    normalization_y = True
    

    ne = ngrid * ngrid
    
    n_train = 4000
    k_max = 16  
    n_layer = 6 
    df = 64
                
    checkpoint_path = f"models/FNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
    
    model = setup_model(in_dim=in_dim, out_dim=out_dim, fc_dim = df, k_max=k_max, n_layer=n_layer, grad_layer=False, dxs=[dx1,dx2], dx_scale=10.0, pad_ratio=0.1, checkpoint_path=checkpoint_path+".pth")
    model = model.to(device)

    x_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(checkpoint_path + "_normalization_x.pth", map_location="cpu", weights_only=True,), device=device)
    y_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(checkpoint_path + "_normalization_y.pth", map_location="cpu", weights_only=True,), device=device)
     
                
    x = torch.from_numpy(x_test[[0],...].astype(np.float32)).to(device)
    # Run one inference, then remove the batch and output-channel axes.
    x = x_normalizer.encode(x)
    y_pred =  model( x ) 
    y_pred = y_normalizer.decode(y_pred)
    y_pred = y_pred.detach().cpu().numpy()[0,...,0]
                    
    np.savez_compressed('data/fno_solver_data.npz', sol=y_pred)
                
            
    return  


if __name__ == "__main__":
    
    # Generate the full cost/accuracy archive used by the plotting script.
    cost, accuracy  = cost_accuracy_fno_solver(downsample_values = [2, 3, 4], k_max_values = [16], n_layer_values = [4, 5, 6], df_values = [64], n_train=4000, n_trial=100)
    # fno_solver(test_index=9999, downsample=2)
