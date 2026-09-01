#!/bin/bash
# Purpose: train the N=1000, 2000, and 8000 MNO data-size sweep.
# Run from: scripts/darcy (all dataset, checkpoint, and log paths are relative).
# Requires: Slurm arrays, the GPU80G partition, one CUDA GPU per task, a
#           preconfigured CUDA Python environment (none is loaded here), enough
#           leading Darcy files plus files 09000--09999, and models/logs dirs.
# Outputs: three model/normalizer checkpoint sets under models/, one log per
#          training size under logs/, and the Slurm stream ParMNO_train.out.
# Example: sbatch mno_train_parallel_ntrain.sh
#SBATCH -o ParMNO_train.out
#SBATCH --qos=low
#SBATCH -J ParMNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-2  

# Map each Slurm array task directly to one training-set size.

N_TRAIN_VALUES=(1000 2000 8000)
INDEX=$SLURM_ARRAY_TASK_ID
N_TRAIN=${N_TRAIN_VALUES[$INDEX]}
K_MAX=16
N_LAYER=6
DF=64
DOWNSAMPLE=2
python mno_train.py \
    --n_train $N_TRAIN \
    --k_max $K_MAX \
    --n_layer $N_LAYER \
    --df $DF \
    --downsample $DOWNSAMPLE \
    > logs/N${N_TRAIN}_k${K_MAX}_nlayer${N_LAYER}_df${DF}_downsample${DOWNSAMPLE}.log
