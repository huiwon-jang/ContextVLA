#!/bin/bash

source ~/anaconda3/bin/activate activate contextvla

export WANDB_API_KEY=42c81e07cdc22dbc10cab806306a5112e1016262

torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=19002 scripts/train_contextvla.py contextvla_robocasa \
    --exp-name=contextvla_robocasa \
    --num-train-steps=30000 \
    --batch-size=16 \
    --accum-steps=2