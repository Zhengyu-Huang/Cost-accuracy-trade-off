#!/bin/bash
#SBATCH -o cpu.out
#SBATCH --qos=low
#SBATCH -J cpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1                # 一个 Julia 进程
#SBATCH --cpus-per-task=16        # 分配 16 个 CPU 核心给该进程
#SBATCH --time=100:00:00

module load anaconda3/2024.10.1 
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl

# generate data
python navier_stokes.py


export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "JULIA_NUM_THREADS = $JULIA_NUM_THREADS"
julia spectral_navier_stokes_solver.jl > spectral_navier_stokes_solver.log
