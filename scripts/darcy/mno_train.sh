#!/bin/bash
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

