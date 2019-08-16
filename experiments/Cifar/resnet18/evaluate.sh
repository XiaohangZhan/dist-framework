#!/bin/bash
work_path=$(dirname $0)
NGPU=8
ITER=$1
python -m torch.distributed.launch --nproc_per_node=$NGPU main.py \
    --config $work_path/config.yaml --launcher pytorch \
    --load-iter $ITER --evaluate
