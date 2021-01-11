#!/bin/bash

lr=1.0
test_batch_size=10000
epochs=40
save_dir='./checkpoint/deterministic'
mode='test'

python examples/main_deterministic_mnist.py --lr=$lr --test-batch-size=$test_batch_size --epochs=$epochs --mode=$mode --save_dir=$save_dir

