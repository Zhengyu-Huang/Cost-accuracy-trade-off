#!/bin/bash
# Purpose: BICMR-cluster variant of the two-GPU, 10k-point M-PCNO training job.
# Run from scripts/aerodynamics:  sbatch mpcno_parallel_train_bicmr.sh
# Requires: the 4000/512 preprocessed 10k archive and the ``fno`` conda
# environment. Outputs checkpoints under models/ and Slurm output under logs/.
# The final detached redirection creates/truncates an extra log file but does
# not capture torchrun output in the checked-in command layout.
#SBATCH -o logs/MPCNO_parallel_train.out
#SBATCH -J MPCNO_parallel_train
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=100:00:00

source activate fno

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
                                    --n_point $N_POINT
                                    > logs/MNO_n_train${N_TRAIN}_k${K_MAX}_nlayer${N_LAYER}_npoint${N_POINT}.log
