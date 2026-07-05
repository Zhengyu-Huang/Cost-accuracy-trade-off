#!/bin/bash
#SBATCH --qos=low
#SBATCH -J decimate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH -o logs/slurm_%a.out
#SBATCH --array=0-14

module load conda
source activate meshlab

INDEX=$SLURM_ARRAY_TASK_ID
# ========== params ==========

N_POINT=10000
BASE_PATH="/lustre/home/2306192137/Cost-accuracy-trade-off/data/aerodynamics/"

DATA_NAME_VALUES=("E_S_WWC_WM" "E_S_WW_WM" "F_D_WM_WW_1" "F_D_WM_WW_2" "F_D_WM_WW_3" "F_D_WM_WW_4" "F_D_WM_WW_5" "F_D_WM_WW_6" "F_D_WM_WW_7" "F_D_WM_WW_8" "F_S_WWC_WM" "F_S_WWS_WM" "N_S_WWC_WM" "N_S_WWS_WM" "N_S_WW_WM")
DATA_NAME=${DATA_NAME_VALUES[$INDEX]}

input_vtk_dir="${BASE_PATH}PressureVTK/${DATA_NAME}"
input_ply_dir="${BASE_PATH}PressurePLY/${DATA_NAME}"
output_ply_dir="${BASE_PATH}PressurePLY_Processed/${DATA_NAME}"
output_vtk_dir="${BASE_PATH}PressureVTK_Processed_${N_POINT}/${DATA_NAME}"

python decimate.py "${input_vtk_dir}" "${input_ply_dir}" "${output_ply_dir}" "${output_vtk_dir}" ${N_POINT}
