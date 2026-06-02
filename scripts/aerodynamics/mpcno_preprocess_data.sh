#!/bin/bash
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

python mpcno_preprocess_data.py --n_each 400   # "cell_centered" , "vertex_centered"
