#!/bin/bash
#SBATCH -o cpu.out
#SBATCH --qos=low
#SBATCH -J cpu
#SBATCH --nodes=1 
#SBATCH --ntasks=4
#SBATCH --time=100:00:00

module load anaconda3/2024.10.1 
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl

. /lustre/home/2306192137/src/venv-firedrake/bin/activate

python multigrid_darcy_solver.py  > logs/multigrid_darcy_solver.log

