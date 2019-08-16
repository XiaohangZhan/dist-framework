#!/bin/bash
work_path=$(dirname $0)
partition=VI_SP_Y_1080TI
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $partition -n8 -x SH-IDC1-10-5-30-62 \
    --gres=gpu:8 --ntasks-per-node=8 \
    python -u main.py \
        --config $work_path/config.yaml --launcher slurm \
        --load-iter 56000 --evaluate
