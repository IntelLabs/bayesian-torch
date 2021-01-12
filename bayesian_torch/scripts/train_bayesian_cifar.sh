#!/bin/bash

model=resnet20
mode='train'
batch_size=128
lr=0.001

python examples/main_bayesian_cifar.py --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size
