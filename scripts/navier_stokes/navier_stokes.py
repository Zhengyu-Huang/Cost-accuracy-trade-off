import os, sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt



# Add the parent directory (project/utility) to Python's search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utility.gaussian_random_fields import gaussian_random_field_2d


def generate_initial_condition(nx = 256, ny = 256, ndata = 10):
    """
    Generate synthetic Navier Stokes flow data: random vorticity fields 
    Saves all initial vorticity condition in navier_stokes_zeta0.npy
    """
    Path("../../data/navier_stokes").mkdir(parents=True, exist_ok=True)
    
    L = 1.0
    
    zeta0_data = gaussian_random_field_2d(ndata, [nx, ny], [L, L], sigma= 7**(3/2), tau = 7.0, alpha = 2.5, bc_name = 'periodic', seed = 42)
        
    np.save(f"../../data/navier_stokes/navier_stokes_zeta0.npy", zeta0_data)


    x, y = np.meshgrid(np.linspace(0,L,nx,endpoint=False), np.linspace(0,1,ny,endpoint=False), indexing='ij')
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        im = axs[i].pcolormesh(x, y, zeta0_data[i,:,:])
        axs[i].set_title("initial vorticity");axs[i].set_aspect('equal')
        fig.colorbar(im, ax=axs[i])
    plt.savefig("initial_vorticity.png")
        

def visualize_data(data_ind = 0):
        
    dt = 1.0
    L = 1.0
    
    data = np.load(f"../../data/navier_stokes/navier_stokes_{data_ind:05d}.npy")
    f_data, zeta_data = data[0,...], data[1:,...]
    
    nsaves, nx, ny = zeta_data.shape
    x, y = np.linspace(0,L,nx,endpoint=False), np.linspace(0,L,ny,endpoint=False)
    
    x_mesh, y_mesh = np.meshgrid(x, y, indexing='ij')
            
    fig, axs = plt.subplots(2, 7, figsize=(21, 12))
    axs = axs.reshape(-1)
    im = axs[0].pcolormesh(x_mesh, y_mesh, f_data)
    axs[0].set_title("vorticity force");axs[0].set_aspect('equal')
    fig.colorbar(im, ax=axs[0])
    
    indices = sorted(set([0, 1, 2] + [(nsaves-1) * k // 10 for k in range(1, 11)]))

    for i,it in enumerate(indices):
        im = axs[i+1].pcolormesh(x_mesh, y_mesh, zeta_data[it,...])
        axs[i+1].set_title(f"vorticity (T=${dt*it:.1f})");axs[i+1].set_aspect('equal')
        fig.colorbar(im, ax=axs[i+1])

    fig.savefig(f"Navier_Stokes_flow_{data_ind}.png")



def cost_accuracy_plot():
    cost_accuracy_traditional_solver_data = np.load('cost_accuracy_traditional_solver_data.npz', allow_pickle=True)   # 注意 allow_pickle=True
    cost = cost_accuracy_traditional_solver_data['cost']
    accuracy = cost_accuracy_traditional_solver_data['accuracy']
    n_downsample = cost_accuracy_traditional_solver_data["sol_length"] 
    sol = [cost_accuracy_traditional_solver_data[f'sol_{i}'] for i in range(n_downsample)]   # list of [F_phys, vorticity_ref, vorticity_sol], where data : nt+1 by n by n array. 
            
    nt2_plus1, ngrid, _ = sol[0].shape
    nt = (nt2_plus1 - 1)//2

    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    x, y = np.meshgrid(np.linspace(0,1,ngrid,endpoint=False), np.linspace(0,1,ngrid,endpoint=False), indexing='ij')
    im = axs[0,0].pcolormesh(x, y, sol[0][nt,...], shading = "gouraud") # vorticity_ref
    fig.colorbar(im, ax=axs[0,0])
    axs[0,0].set_title(fr'$u ({ngrid} \times {ngrid})$')
    im = axs[1,0].pcolormesh(x, y, sol[0][0,...], shading = "gouraud")  # g_ref
    fig.colorbar(im, ax=axs[1,0])
    axs[1,0].set_title(r'$g$')
    for downsample in range(1,4):
        stride = 2**downsample
        im = axs[0,downsample].pcolormesh(x[0::stride, 0::stride], y[0::stride, 0::stride], sol[downsample][2*nt,...], shading = "gouraud") # vorticity_sol
        fig.colorbar(im, ax=axs[0,downsample])
        axs[0,downsample].set_title(fr'$u ({(ngrid)//stride} \times {(ngrid)//stride})$')
        im = axs[1,downsample].pcolormesh(x[0::stride, 0::stride], y[0::stride, 0::stride], np.fabs(sol[downsample][2*nt,...] - sol[downsample][nt,...]), shading = "gouraud") # vorticity_error
        fig.colorbar(im, ax=axs[1,downsample])
        axs[1,downsample].set_title('Error')
    fig.tight_layout()
    fig.savefig("solution_traditional_solver.pdf")    
    
    
    
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    mean_cost = np.mean(cost, axis=1)     
    mean_accuracy = np.mean(accuracy, axis=1)    
    std_cost  = np.std(cost, axis=1, ddof=1)       
    std_accuracy  = np.std(accuracy, axis=1, ddof=1)

    axs[0].loglog(mean_accuracy, mean_cost[...,0], 'o-')
    axs[0].errorbar(mean_accuracy, mean_cost[...,0], xerr=std_accuracy, fmt='o')
    axs[0].set_xlabel("Rel. error")
    axs[0].set_ylabel("Floating-point cost")
    axs[1].loglog(mean_accuracy, mean_cost[...,1], 'o-')
    axs[1].errorbar(mean_accuracy, mean_cost[...,1], xerr=std_accuracy, yerr=std_cost[...,1], fmt='o')
    axs[1].set_xlabel("Rel. error")
    axs[1].set_ylabel("CPU cost (s)")

    fig.tight_layout()
    fig.savefig("cost_accuracy_traditional_solver.pdf")    
        


# Example usage
if __name__ == "__main__":
    # generate_initial_condition(nx = 256, ny = 256, ndata = 2000)
    # visualize_data(data_ind = 0)
    # visualize_data(data_ind = 1)
    cost_accuracy_plot()