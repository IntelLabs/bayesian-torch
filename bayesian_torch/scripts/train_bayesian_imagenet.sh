#!/bin/bash

model=resnet50
mode='train'
batch_size=128
lr=0.001
moped=True
delta=0.5

python -u examples/main_bayesian_imagenet.py data/imagenet --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --moped=$moped --delta=$delta
