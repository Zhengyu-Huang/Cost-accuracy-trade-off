#!/bin/bash
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

