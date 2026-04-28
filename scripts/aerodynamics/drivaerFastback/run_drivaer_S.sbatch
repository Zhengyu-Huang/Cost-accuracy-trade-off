#!/bin/bash


#SBATCH -o drivaer_of12S
#SBATCH --qos=low
#SBATCH -J drivaer_of12S
#SBATCH --nodes=1 
#SBATCH --ntasks=8
#BATCH --time=12:00:00


module load OpenFoam/12-openmpi4.1.5-gcc12.2.0

export OMP_NUM_THREADS=1

./Allclean || true

./Allrun -m S -cores 8


