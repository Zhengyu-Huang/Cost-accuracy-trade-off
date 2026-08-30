#!/bin/bash
# Purpose: export error arrays, representative VTK predictions, and histograms.
# Run from scripts/aerodynamics:  sbatch mpcno_plot_results.sh
# Requires: the 20k/40k 4000/512 archives and matching trained checkpoints plus
# output normalizers. The active Python entry point processes 20k and 40k only.
# Outputs are written under data/ and figs/; Slurm output is stored in logs/.
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
