#!/bin/bash

model=resnet20
mode='test'
batch_size=100
num_monte_carlo=50

python examples/main_bayesian_flipout_cifar.py --arch=$model --mode=$mode --batch-size=$batch_size --num_monte_carlo=$num_monte_carlo
