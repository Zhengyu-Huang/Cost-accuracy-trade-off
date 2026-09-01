#!/bin/bash
# Purpose: train the 3-layer-count by 3-grid standard FNO configuration sweep.
# Run from: scripts/darcy_fno (dataset, checkpoint, and log paths are relative).
# Requires: Slurm arrays, the GPU80G partition, one CUDA GPU per task, a
#           preconfigured CUDA Python environment (none is loaded here), Darcy
#           files 00000--03999 and 09000--09999, and models/logs directories.
# Outputs: nine model/normalizer checkpoint sets under models/, one log per
#          configuration under logs/, and the Slurm stream ParFNO_train.out.
# Example: sbatch fno_train_parallel.sh
#SBATCH -o ParFNO_train.out
#SBATCH --qos=low
#SBATCH -J ParFNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-8  

# Parameter grid flattened across the Slurm array index.

N_TRAIN=4000


K_MAX_VALUES=(16)
N_LAYER_VALUES=(4 5 6)
DF_VALUES=(64)
DOWNSAMPLE_VALUES=(2 3 4)

K_MAX_COUNT=${#K_MAX_VALUES[@]}
N_LAYER_COUNT=${#N_LAYER_VALUES[@]}
DF_COUNT=${#DF_VALUES[@]}
DOWNSAMPLE_COUNT=${#DOWNSAMPLE_VALUES[@]}

INDEX=$SLURM_ARRAY_TASK_ID




DOWNSAMPLE_INDEX=$((INDEX % DOWNSAMPLE_COUNT))
TMP_INDEX=$((INDEX / DOWNSAMPLE_COUNT))

DF_INDEX=$((TMP_INDEX % DF_COUNT))
TMP_INDEX=$((TMP_INDEX / DF_COUNT))

N_LAYER_INDEX=$((TMP_INDEX  % N_LAYER_COUNT))
TMP_INDEX=$((TMP_INDEX  / N_LAYER_COUNT))

K_MAX_INDEX=$((TMP_INDEX  % K_MAX_COUNT))



K_MAX=${K_MAX_VALUES[$K_MAX_INDEX]}
N_LAYER=${N_LAYER_VALUES[$N_LAYER_INDEX]}
DF=${DF_VALUES[$DF_INDEX]}
DOWNSAMPLE=${DOWNSAMPLE_VALUES[$DOWNSAMPLE_INDEX]}


python fno_train.py \
    --n_train $N_TRAIN \
    --k_max $K_MAX \
    --n_layer $N_LAYER \
    --df $DF \
    --downsample $DOWNSAMPLE \
    > logs/N${N_TRAIN}_k${K_MAX}_nlayer${N_LAYER}_df${DF}_downsample${DOWNSAMPLE}.log








