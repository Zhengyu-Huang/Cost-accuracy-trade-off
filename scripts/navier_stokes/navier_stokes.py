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


    x, y = np.meshgrid(np.linspace(0,L,nx,endpoint=False), np.linspace(0,1,ny,endpoint=False), indexing='xy')
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
    
    nsaves, ngrid, _ = zeta_data.shape
    x, y = np.linspace(0,L,ngrid,endpoint=False), np.linspace(0,L,ngrid,endpoint=False)
    

    x_mesh, y_mesh = np.meshgrid(x, y, indexing='xy')
            
    
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
    
# Example usage
if __name__ == "__main__":
    # generate_initial_condition(nx = 256, ny = 256, ndata = 10)
    visualize_data(data_ind = 0)
    visualize_data(data_ind = 1)