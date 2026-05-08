import sys
import os
import math 
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mno_train import preprocess_data, load_test_data

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
# 添加到路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from nn.mno import MNO2d, MNO_train, setup_model, mno_floating_point_cost
from utility.normalizer import UnitGaussianNormalizer


np.set_printoptions(precision=10, suppress=True)



def mno_solve(model, x_normalizer, y_normalizer, x, nt, device):
    """
    Neural operator solver for time-dependent PDEs using a learned operator.
    
    Args:
        model: Neural operator model that advances the solution by one time step.
               It takes a state tensor of shape (batch_size, nx, ny, in_channels)
               and returns the next state of shape (batch_size, nx, ny, out_dim).
        x_normalizer: Normalizer for input states (e.g., mean/std scaling).
                      If None, no normalization is applied.
        y_normalizer: Normalizer for output states (e.g., mean/std scaling).
                      If None, no denormalization is applied.
        device: Torch device on which computations are performed (e.g., 'cuda' or 'cpu').
        x: Initial condition tensor of shape (batch_size, nx, ny, in_channels).
           The first `out_dim` channels (default 1) are used as the initial solution.
        nt: Number of time steps to roll out.
    
    Returns:
        y_pred: Predicted solution over time, numpy array of shape
                (batch_size, nt+1, nx, ny, out_dim).
    """
    
    batch_size, nx, ny, in_dim = x.shape
    
    L = 1.0
    out_dim = 1 
   
    model = model.to(device)

    normalization_x = x_normalizer is not None
    normalization_y = y_normalizer is not None


    if normalization_x:
        x_normalizer.to(device)
    if normalization_y:
        y_normalizer.to(device)

    x = torch.from_numpy(x.astype(np.float32)).to(device)
    
    # Initialize prediction array: (batch, time, nx, ny, out_dim)
    y_pred = torch.zeros((batch_size, nt+1, nx, ny, out_dim), device=device)

    # initialize solution at 0
    y_pred[:, 0, ...] = x[... , :out_dim].clone()
    
    start_time = time.perf_counter()
    for i in range(nt):
        x[..., :out_dim] = y_pred[:, i ,...]
        y_pred[:, i+1 ,...] =  model( x = (x_normalizer.encode(x) if normalization_x else x) ) 
        if normalization_y:
            y_pred[:, i+1, ...] = y_normalizer.decode(y_pred[:, i+1, ...])
    end_time = time.perf_counter()        
    
    y_pred = y_pred.detach().cpu().numpy()
    solve_time = end_time - start_time

    return y_pred, solve_time




def mno_solve_visualize(n_layer, df, downsample, k_max, n_train):
    """
    Neural operator solver error .
    """
    # load reference solution
    nt = 50
    nx = ny = 256
    L = 1.0
    nx, ny = nx//(2**downsample), ny//(2**downsample)
    dim, in_dim, out_dim = 2, 4, 1
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = f"models/MNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
          
          
    # x_test is [batch_size , nt+1 , nx , ny , 4]
    batch_size = 5
    x_test, dx1, dx2 = load_test_data(batch_size, nt, downsample)

    
    y_ref = x_test[...,:out_dim] #[batch_size , nt+1 , nx , ny , out_dim] 
    

    
    model = setup_model(in_dim=in_dim, out_dim=out_dim, fc_dim=df, k_max=k_max, n_layer=n_layer, dxs=[dx1,dx2], dx_scale=10.0, pad_ratio=0, incremental = True, checkpoint_path=checkpoint_path+".pth")
    model = model.to(device)
    
    normalization_x = False
    normalization_y = False
    
    x_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(checkpoint_path + "_normalization_x.pth", map_location="cpu", weights_only=True,), device=device) if normalization_x else None
    y_normalizer = UnitGaussianNormalizer.from_state_dict(torch.load(checkpoint_path + "_normalization_y.pth", map_location="cpu", weights_only=True,), device=device) if normalization_y else None
        

    x = x_test[:, 0,...] #[batch_size , nx , ny , in_dim] 
    
    y_pred, _ = mno_solve(model, x_normalizer, y_normalizer, x, nt, device)
    
    
    ################### Postprocessing ######################
    print(y_pred.shape, y_ref.shape)
    error = np.zeros((batch_size, nt+1))
    rel_error = np.zeros((batch_size, nt+1))
    for bs in range(batch_size):
        for i in range(nt+1):
            error[bs,i] = np.linalg.norm(y_ref[bs,i,...] - y_pred[bs,i,...]) * np.sqrt(dx1*dx2)
            rel_error[bs,i] = np.linalg.norm(y_ref[bs,i,...] - y_pred[bs,i,...])/(np.linalg.norm(y_ref[bs,i,...]))

    fig, axs = plt.subplots(1,1, figsize=(6, 6))
    axs.plot(np.mean(error, axis=0), "-o", markerfacecolor="none", label = "abs. error")
    axs.plot(np.mean(rel_error, axis=0), "-o", markerfacecolor="none", label = "rel. error")
    axs.legend()
    print("mean rel_error = ", np.mean(rel_error, axis=0))
    plt.savefig("MNO_error.pdf")
    
    # visualize the first test
    y_ref, y_pred = y_ref[0,...], y_pred[0,...]
    x1, x2 = np.linspace(0,L,nx,endpoint=False), np.linspace(0,L,ny,endpoint=False)
    
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing='ij')
    ############ Visualization ########################

    fig, axs = plt.subplots(2, 5, figsize=(16, 8))
    for i in range(5):
        im = axs[0, i].pcolormesh(x1_grid, x2_grid, y_ref[i*10,...,0])
        fig.colorbar(im, ax=axs[0, i])
        axs[0, i].set_aspect('equal')
        axs[0, i].set_title(f"vorticity (T=${i*10})");
        
        im = axs[1, i].pcolormesh(x1_grid, x2_grid, y_pred[i*10,...,0])
        fig.colorbar(im, ax=axs[1, i])
        axs[1, i].set_aspect('equal')
    
        
    plt.savefig("figs/MNO_prediction.png")




def cost_accuracy_mno_solver_helper(device, downsample, k_max_values, n_layer_values, df_values, n_train, n_trial):
    """
    Neural operator solver error .
    """
    # load reference solution
    
    nt = 50
    nx = ny = 256
    nx, ny = nx//(2**downsample), ny//(2**downsample)
    ne = nx*ny
    L = 1.0
    
    cost = np.zeros((len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2))
    accuracy = np.zeros((len(k_max_values), len(n_layer_values), len(df_values), n_trial, nt+1))
    
    dim, in_dim, out_dim = 2, 4, 1
    
    x_test, dx1, dx2 = load_test_data(n_trial, nt, downsample)

    
    normalization_x = False
    normalization_y = False
    x_normalizer, y_normalizer = None, None 

    
    n_repeat = 10
    for k_max_index, k_max in enumerate(k_max_values):
        for n_layer_index, n_layer in enumerate(n_layer_values):
            for df_index, df in enumerate(df_values):
                
                checkpoint_path = f"models/MNO_model_N{n_train}_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}"
                # checkpoint_path = f"models/MNO_model_N4000_k{k_max}_nlayer{n_layer}_df{df}_downsample{downsample}.pth"
    
                model = setup_model(in_dim=in_dim, out_dim=out_dim, k_max=k_max, n_layer=n_layer, df=df, dx1=dx1, dx2=dx2, dx_scale=10.0, device=device, checkpoint_path=checkpoint_path)
                model = model.to(device)

                for i in range(n_trial):
                    x = x_test[[i], 0, ...]            #[batch_size , nx , ny , in_dim] 
                    y_ref = x_test[[i], ..., :out_dim] #[batch_size , nt+1 , nx , ny , out_dim]
                    # Warm up
                    y_pred, sol_time = mno_solve(model, x_normalizer, y_normalizer, x, nt, device)
                    
                    sol_time_ave = 0
                    for j in range(n_repeat):
                        y_pred, sol_time = mno_solve(model, x_normalizer, y_normalizer, x, nt, device)
                        sol_time_ave += sol_time
                    sol_time_ave /= n_repeat
                    
                    error = np.zeros(nt+1)
                    rel_error = np.zeros(nt+1)
                    for i in range(nt+1):
                        error[i] = np.linalg.norm(y_ref[i,...] - y_pred[i,...]) * np.sqrt(dx1*dx2)
                        rel_error[i] = np.linalg.norm(y_ref[i,...] - y_pred[i,...])/(np.linalg.norm(y_ref[i,:]))

                    cost[k_max_index, n_layer_index, df_index, i, 0] = mno_floating_point_cost(dim, in_dim, out_dim, k_max, df, n_layer, ne, mesh_type="structured")
                    cost[k_max_index, n_layer_index, df_index, i, 1] = sol_time_ave
                    accuracy[k_max_index, n_layer_index, df_index, i,:] = rel_error
                    
                    print("relative error is : ", rel_error, " flops = ", cost[k_max_index, n_layer_index, df_index, i, 0], " computing_time = ", sol_time_ave)

    
    return  cost, accuracy




def cost_accuracy_mno_solver(downsample_values, k_max_values, n_layer_values, df_values, n_train, n_trial = 10):
    cost     = np.zeros((len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 3))
    accuracy = np.zeros((len(downsample_values), len(k_max_values), len(n_layer_values), len(df_values), n_trial, 2, nt+1))
    
    for downsample_index, downsample in enumerate(downsample_values):
        for device in [torch.device('cuda') , torch.device('cpu')]:
        # for device in [torch.device('cuda') , torch.device('cpu')]:
            cost_ds, accuracy_ds = cost_accuracy_mno_solver_helper(device, downsample, k_max_values = k_max_values, n_layer_values = n_layer_values, df_values = df_values, n_train = n_train, n_trial = n_trial)
            cost[downsample_index, :, :, :, :, 0] = cost_ds[...,0]
            if device.type == 'cpu':
                cost[downsample_index, :, :, :,:, 1] = cost_ds[...,1]
                accuracy[downsample_index, ..., 0,:] = accuracy_ds
            else:
                cost[downsample_index, :, :, :,:, 2] = cost_ds[...,1]
                accuracy[downsample_index, ..., 1,:] = accuracy_ds
                
            
            
    np.savez_compressed('data/cost_accuracy_mno_solver_data.npz', cost=cost, accuracy=accuracy)

    return  cost, accuracy 

if __name__ == "__main__":
    
    
    ###################################
    # load parameters
    ###################################

    # cost, accuracy, sol  = cost_accuracy_mno_solver(downsample_values = [2, 3], k_max_values = [16, 32], n_layer_values = [4, 5], df_values = [32, 64, 128])
    # cost, accuracy, sol  = cost_accuracy_mno_solver(downsample_values = [2], k_max_values = [16], n_layer_values = [5], df_values = [128])
    
    mno_solve_visualize(n_layer=6, df=64, downsample=1, k_max=16, n_train=10000)