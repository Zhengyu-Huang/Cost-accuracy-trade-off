#!/bin/bash
# Run the currently selected Python legacy plotter, then the active Julia solver.
# Submit from scripts/navier_stokes with: sbatch bash_cpu.sh
# Prerequisites: the site-specific modules below,
# cost_accuracy_traditional_solver_data.npz in this directory,
# ../../data/navier_stokes/navier_stokes_01999.npy, and data/.
# Outputs: cpu.out, the legacy PDFs, spectral_navier_stokes_solver.log, and
# data/traditional_solver_data.npz. The script does not currently generate data
# because both source files have different active bottom-of-file drivers.
#SBATCH -o cpu.out
#SBATCH --qos=low
#SBATCH -J cpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1                # One Julia process
#SBATCH --cpus-per-task=16        # Sixteen CPU cores for that process
#SBATCH --time=100:00:00

module load anaconda3/2024.10.1 
module load gcc/12.2.0
module load openmpi/4.1.5-gcc_12.2.0
module load cmake/3.31.9
module load OpenBLAS/0.3.17


export OMPI_MCA_btl=self,vader,tcp
unset OMPI_MCA_pml
unset OMPI_MCA_mtl

# generate initial condition
python navier_stokes.py

# Run the currently active julia driver to generate data
# (generate_data(nx = 256, ny = 256, ndata = 2000)).
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "JULIA_NUM_THREADS = $JULIA_NUM_THREADS"
julia spectral_navier_stokes_solver.jl > spectral_navier_stokes_solver.log
