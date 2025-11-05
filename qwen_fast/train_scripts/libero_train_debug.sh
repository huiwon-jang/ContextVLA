#!/bin/bash

source ~/anaconda3/bin/activate activate contextvla

python scripts/train_contextvla.py contextvla_libero \
    --exp-name=debug \
    --num-train-steps=60000 \
    --batch-size=2 \
    --num-workers=0