## Install firedrake on wm2


module load anaconda3/2024.10.1 
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl



curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure

git clone --branch $(python3 firedrake-configure --no-package-manager --show-petsc-version) https://gitlab.com/petsc/petsc.git
cd petsc/

python3 ../firedrake-configure --no-package-manager --show-petsc-configure-options | xargs -L1 ./configure
make PETSC_DIR=/lustre/home/2306192137/src/petsc PETSC_ARCH=arch-firedrake-default all
make PETSC_DIR=/lustre/home/2306192137/src/petsc PETSC_ARCH=arch-firedrake-default check
cd ..

python3 -m venv venv-firedrake
. venv-firedrake/bin/activate

pip cache purge

export $(python3 firedrake-configure --no-package-manager  --show-env)
pip install --no-binary h5py 'firedrake[check]'

firedrake-check

! When any library is not supported, for example h5py
pip uninstall -y h5py
pip install --upgrade firedrake




## Use firedrake on wm2

module load anaconda3/2024.10.1 
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl

. /lustre/home/2306192137/src/venv-firedrake/bin/activate



## Generate training data on wm2

sbatch bash_cpu.sh

Generate data in `../../data/darcy/darcy_data_{i:05d}.npy`

Each data file contains a `(n+1) by (n+1) by 2` numpy array: 

2D array of shape (nx+1, ny+1), where arr[i, j] corresponds to the grid point
(x = i/nx, y = j/ny). The mesh must have vertices exactly at these points.

These two channels correspond to the permeability field `kappa` and pressure solution `u` 

To visualize:
    x, y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny), indexing='ij')
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    im = axs[0].pcolormesh(x, y, kappa)


## Numerical method cost accuracy trade-off
Estimate the cost (floating point flops and CPU time) and accuracy (relative error) with different downsampled meshes, save the data
cost_accuracy_traditional_solver() in multigrid_darcy_solver.py

Plot solutions with different downsampled meshes, plot cost accuracy trade-off curves 
cost_accuracy_plot() in multigrid_darcy_solver.py


## Neural Operator method cost accuracy trade-off
### Training neural operator
```bash
    sbatch mno_train_parallel.py
```

### Cost accuracy trade-off
Estimate the cost (floating point flops and CPU/GPU time) and accuracy (relative error) with different downsampled meshes, save the data
cost_accuracy_neural_operator() in mno_darcy_solver.py

Plot solutions with different downsampled meshes, plot cost accuracy trade-off curves 
cost_accuracy_plot() in multigrid_darcy_solver.py

