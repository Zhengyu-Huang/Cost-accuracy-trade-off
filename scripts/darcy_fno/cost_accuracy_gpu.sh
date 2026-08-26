#!/bin/bash
# Purpose: benchmark all configured FNO checkpoints on both GPU and CPU.
# Run from: scripts/darcy_fno (all model, data, and log paths are relative).
# Requires: Slurm, the GPU80G partition, one CUDA GPU, the conda module, the
#           final 100 Darcy samples, nine FNO checkpoints/normalizers, and
#           existing data/logs directories.
# Outputs: data/cost_accuracy_fno_solver_data.npz, logs/fno_darcy_solver.log,
#          and the Slurm stream FNO_train.out.
# Example: sbatch cost_accuracy_gpu.sh
#SBATCH -o FNO_train.out
#SBATCH --qos=low
#SBATCH -J FNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
python fno_darcy_solver.py > logs/fno_darcy_solver.log
