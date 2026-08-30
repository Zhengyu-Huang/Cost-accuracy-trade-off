#!/bin/bash
# Purpose: preprocess decimated 10k surfaces into padded M-PCNO mesh tensors.
# Run from scripts/aerodynamics:  sbatch mpcno_preprocess_data.sh
# Requires: all category directories below PressureVTK_Processed_10000 and the
# ``pytorch`` environment. The active command loads at most 400 cases/category.
# Outputs mpcno_data.npz and mpcno_data_names_list.npy in that data directory.
#SBATCH -o logs/Preprocess_data_vertex_centered.out
#SBATCH --qos=low
#SBATCH -p C064M0256G
##SBATCH -p C064M1024G
#SBATCH -J Preprocess_data_vertex_centered
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=100:00:00

module load conda
source activate pytorch

python mpcno_preprocess_data.py --n_each 400 --n_point 10000  
