#!/bin/bash

model=resnet50
mode='test'
val_batch_size=100
num_monte_carlo=50

python examples/main_bayesian_imagenet.py data/imagenet --arch=$model --mode=$mode --val_batch_size=$val_batch_size --num_monte_carlo=$num_monte_carlo
