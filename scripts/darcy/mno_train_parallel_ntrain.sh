#!/bin/bash
#SBATCH -o ParMNO_train.out
#SBATCH --qos=low
#SBATCH -J ParMNO_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --array=0-2  

# ========== params ==========

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
