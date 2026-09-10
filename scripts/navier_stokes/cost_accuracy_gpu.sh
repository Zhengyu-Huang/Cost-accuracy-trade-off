#!/bin/bash
#SBATCH -o cost_accuracy.out
#SBATCH --qos=normal
#SBATCH -J cost_accuracy
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
# python mno_navier_stokes_solver.py > logs/mno_navier_stokes_solver.log
julia spectral_navier_stokes_solver.jl > logs/spectral_navier_stokes_solver.log
