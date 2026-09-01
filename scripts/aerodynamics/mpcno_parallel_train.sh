#!/bin/bash
# Purpose: train the 10k-point M-PCNO configuration on two GPUs using DDP.
# Run from scripts/aerodynamics:  sbatch mpcno_parallel_train.sh
# Requires: the 4000/512 preprocessed 10k archive and the cluster ``pytorch``
# environment. Outputs periodic/final checkpoints under models/ and Slurm output
# under logs/. As written, the final detached redirection creates/truncates its
# extra log file but does not capture torchrun output.
#SBATCH -o logs/MPCNO_parallel_train.out
#SBATCH --qos=normal
#SBATCH -J MPCNO_parallel_train
#SBATCH -p GPU80G
#SBATCH --nodes=1 
#SBATCH --ntasks=32
#SBATCH --gres=gpu:2
#SBATCH --time=100:00:00

module load conda
source activate pytorch

export MASTER_ADDR=$(hostname)   # Single-node rendezvous address.
export MASTER_PORT=29504         # Rendezvous port; change if already occupied.
export NCCL_DEBUG=INFO           # Emit NCCL communication diagnostics.

echo "Starting distributed training on $(hostname)"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"


N_TRAIN=4000
K_MAX=16
N_LAYER=4
N_POINT=10000

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                                    mpcno_parallel_train.py \
                                    --grad True \
                                    --geo True \
                                    --geointegral True \
                                    --n_layer $N_LAYER \
                                    --k_max $K_MAX \
                                    --batch_size 4 \
                                    --epochs 200 \
                                    --n_train $N_TRAIN \
                                    --n_test 512 \
                                    --dx_scale 10.0 \
                                    --n_point $N_POINT \
                                    > logs/MPCNO_n_train${N_TRAIN}_k${K_MAX}_nlayer${N_LAYER}_npoint${N_POINT}.log
