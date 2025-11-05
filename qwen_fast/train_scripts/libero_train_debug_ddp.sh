#!/bin/bash

source ~/anaconda3/bin/activate activate contextvla

torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=19001 scripts/train_contextvla.py contextvla_libero \
    --exp-name=debug \
    --num-train-steps=30000 \
    --batch-size=8 \
    --num-workers=0 \
    --accum-steps=2