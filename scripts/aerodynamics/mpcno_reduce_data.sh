#!/bin/bash
# Purpose: select the seeded 4000/512 subset from the full 10k archive.
# Run from scripts/aerodynamics:  sbatch mpcno_reduce_data.sh
# Requires: PressureVTK_Processed_10000/mpcno_data.npz and its names array.
# Outputs aligned n_train4000_n_test512 data and names archives in the same
# directory; the active launcher does not process the 20k or 40k archives.
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
