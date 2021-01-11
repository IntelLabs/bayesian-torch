#!/bin/bash

lr=1.0
batch_size=256
epochs=40
save_dir='./checkpoint/deterministic'
mode='train'

python examples/main_deterministic_mnist.py --lr=$lr --batch-size=$batch_size --epochs=$epochs --mode=$mode --save_dir=$save_dir

