#!/bin/bash
#SBATCH -o logs/MPCNO_reduce_data_vertex_centered.out
#SBATCH --qos=low
#SBATCH -p C064M1024G
#SBATCH -J PCNO_reduce_data_vertex_centered
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=100:00:00

module load conda
source activate pytorch
python mpcno_reduce_data.py --n_train 4000 --n_test 512 --n_point 10000
