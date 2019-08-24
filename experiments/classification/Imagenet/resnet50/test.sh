#!/bin/bash
work_path=$(dirname $0)
ITER=$1
python scripts/test_classification.py \
    --config $work_path/config.yaml \
    --load-iter $ITER
