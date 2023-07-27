#!/bin/bash

model=resnet20
mode='ptq'
batch_size=1000
num_monte_carlo=20
checkpoint_file=./checkpoint/bayesian/bayesian_resnet20_cifar.pth

python examples/main_bayesian_cifar_dnn2bnn.py --arch=$model --mode=$mode --batch-size=$batch_size --num_monte_carlo=$num_monte_carlo --model-checkpoint=$checkpoint_file
