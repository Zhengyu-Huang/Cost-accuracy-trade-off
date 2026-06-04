#!/bin/bash
#SBATCH --qos=low
#SBATCH -J decimate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH -o logs/decimate.out


python decimate.py "data/openfoam_L.vtk" "data/openfoam_L_decimate.vtk" 20000
