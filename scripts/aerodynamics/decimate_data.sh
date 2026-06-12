#!/bin/bash
#SBATCH --qos=low
#SBATCH -J decimate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH -o logs/decimate.out

module load conda
source activate meshlab
python decimate.py "/lustre/home/2306192137/OpenFOAM/drivaerFastback_L/postProcessing/car/4000/patch.vtk" "data/openfoam_L_decimate.vtk" 20000
