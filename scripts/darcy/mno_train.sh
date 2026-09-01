#!/bin/bash
# Purpose: train the fixed N=8000, k_max=4, layers=4, width=16 MNO at stride 4.
# Run from: scripts/darcy (all dataset, checkpoint, and log paths are relative).
# Requires: Slurm, the GPU80G partition, one CUDA GPU, the conda module, Darcy
#           files 00000--07999 and 09000--09999, and models/logs directories.
# Outputs: the matching model/normalizer files under models/, the named log
#          under logs/, and the Slurm stream MNO_train.out.
# Example: sbatch mno_train.sh
#SBATCH -o MNO_train.out
#SBATCH --qos=low
#SBATCH -J MNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00

module load conda



N_TRAIN=8000
K_MAX=4
N_LAYER=4
DF=16
DOWNSAMPLE=2
python mno_train.py --n_train $N_TRAIN \
    --k_max $K_MAX \
    --n_layer $N_LAYER \
    --df $DF \
    --downsample $DOWNSAMPLE \
    > logs/N${N_TRAIN}_k${K_MAX}_nlayer${N_LAYER}_df${DF}_downsample${DOWNSAMPLE}.log
