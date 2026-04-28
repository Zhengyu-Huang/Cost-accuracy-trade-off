#!/bin/bash

#SBATCH -o drivaer_of12S_postprocess
#SBATCH --qos=low
#SBATCH -J drivaer_of12S_postprocess
#SBATCH --nodes=1 
#SBATCH --ntasks=8
#BATCH --time=12:00:00


module load OpenFoam/12-openmpi4.1.5-gcc12.2.0

export OMP_NUM_THREADS=1

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

# Run post-processing to compute Cp and wall shear stress (on patch) data in parallel
runParallel foamPostProcess -solver incompressibleFluid -dict system/controlDict.post

# Reconstruct the mesh
reconstructParMesh -constant

# Reconstruct the fluid data
reconstructPar

# Export vehicle surface data to VTK
foamToVTK -useTimeName -noInternal -allPatches -excludePatches '(inlet freestream ground)' -fields "(p Cp wallShearStress)"
mv VTK/allPatches VTK_car_surface

# Export volume flow-field data to VTK
foamToVTK -useTimeName -fields "(U p)"

