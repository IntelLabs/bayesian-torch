#!/bin/bash

model=resnet50
mode='test'
val_batch_size=100

python examples/main_deterministic_imagenet.py data/imagenet --arch=$model --mode=$mode --val_batch_size=$val_batch_size
