#!/bin/bash
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

export MASTER_ADDR=$(hostname)   # 主节点地址
export MASTER_PORT=29504         # 主节点端口
export NCCL_DEBUG=INFO           # 可选：查看NCCL通信信息

echo "Starting distributed training on $(hostname)"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"


torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
                                    mpcno_parallel_train.py \
                                    --grad True \
                                    --geo True \
                                    --geointegral True \
                                    --n_layer 6 \
                                    --k_max 16 \
                                    --batch_size 4 \
                                    --epochs 500 \
                                    --n_train 2000 \
                                    --n_test 1000 \
                                    --dx_scale 10.0 \
                                    > logs/MPCNO_n_train2000_nlayer6.log
