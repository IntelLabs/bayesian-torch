#!/bin/bash

model=resnet50
mode='test'
val_batch_size=1
num_monte_carlo=1

python examples/main_bayesian_imagenet_bnn2qbnn.py --mode=$mode --val_batch_size=$val_batch_size --num_monte_carlo=$num_monte_carlo ../../datasets