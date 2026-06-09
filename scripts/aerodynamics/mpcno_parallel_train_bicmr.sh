#!/bin/bash
#SBATCH -o logs/MPCNO_parallel_train.out
#SBATCH -J MPCNO_parallel_train
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=100:00:00

source activate fno

export MASTER_ADDR=$(hostname)   # 主节点地址
export MASTER_PORT=29504         # 主节点端口
export NCCL_DEBUG=INFO           # 可选：查看NCCL通信信息

echo "Starting distributed training on $(hostname)"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

 
N_TRAIN=4000
K_MAX=16
N_LAYER=4


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
                                    > logs/MPCNO_n_train${N_TRAIN}_k${K_MAX}_nlayer${N_LAYER}.log
