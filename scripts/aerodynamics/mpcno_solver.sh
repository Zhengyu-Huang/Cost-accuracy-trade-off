#!/bin/bash
#SBATCH -o logs/MPCNO_solver.out
#SBATCH --qos=normal
#SBATCH -J MPCNO_solver
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda
source activate pytorch

python  mpcno_solver.py