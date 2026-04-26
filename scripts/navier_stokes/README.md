## Install GeophysicalFlows on wm2


install Julia

using Pkg
Pkg.add("GeophysicalFlows")



## Generate training data on wm2

sbatch bash_cpu.sh

Generate data in `../../data/navier_stokes/navier_stokes_%05d.npy`

Each data file contains a `nt+2 by n by n` numpy array: 

2D array of shape (nx, ny), where arr[i, j] corresponds to the grid point
(x = i/nx, y = j/ny) with 0<=i<nx , 0<=j<ny. The mesh must have vertices exactly at these points.

These nt+2 channels correspond to the vorticity forcing field `f` and vorticity solution `w_0`,`w_1`,...,`w_nt`  

To visualize:
    x, y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny), indexing='ij')
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    im = axs[0].pcolormesh(x, y, f)
