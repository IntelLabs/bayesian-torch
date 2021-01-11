#!/bin/bash

mode='train'
batch_size=512
num_monte_carlo=50
epochs = 50

python examples/main_flipout_mnist.py --test-batch-size $batch_size --mode $mode --batch-size $batch_size --epochs $epochs