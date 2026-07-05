#!/bin/bash
#SBATCH -o logs/MPCNO_plot_results.out
#SBATCH --qos=low
#SBATCH -p C064M1024G
#SBATCH -J PCNO_plot_results
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=100:00:00

module load conda
source activate pytorch
python mpcno_plot_results.py 
