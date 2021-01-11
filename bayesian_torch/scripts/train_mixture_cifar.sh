#!/bin/bash

model=resnet20
mode='train'
batch_size=512
num_monte_carlo=50

python examples/main_mixture_cifar.py --arch=$model --mode=$mode --batch-size=$batch_size --num_monte_carlo=$num_monte_carlo
