#!/bin/bash
# Purpose: decimate one hard-coded OpenFOAM surface to approximately 20k points.
# Run from scripts/aerodynamics:  sbatch decimate_data.sh
# Requires: the source patch.vtk path below and a conda environment with VTK,
# PyMeshLab, NumPy, and SciPy. The VTK file must contain nodal pressure ``p``.
# Outputs: data/openfoam_L_decimate.vtk, adjacent intermediate PLY files, and
# logs/decimate.out. The active command processes only the listed 4000-time case.
#SBATCH --qos=low
#SBATCH -J decimate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH -o logs/decimate.out

module load conda
source activate meshlab
python decimate.py "/lustre/home/2306192137/OpenFOAM/drivaerFastback_L/postProcessing/car/4000/patch.vtk" "data/openfoam_L_decimate.vtk" 20000
