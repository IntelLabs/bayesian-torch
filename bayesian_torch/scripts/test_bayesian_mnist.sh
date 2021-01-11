#!/bin/bash

test_batch_size=10000
mode='test'
save_dir='./checkpoint/bayesian'
num_monte_carlo=20

python examples/main_bayesian_mnist.py --test-batch-size=$test_batch_size --mode=$mode --save_dir=$save_dir --num_monte_carlo=$num_monte_carlo

