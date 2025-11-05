#!/bin/bash

source ~/anaconda3/bin/activate activate contextvla

torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=19002 scripts/train_contextvla.py contextvla_bridge \
    --exp-name=contextvla_bridge \
    --num-train-steps=30000 \
    --batch-size=16 \
    --accum-steps=2