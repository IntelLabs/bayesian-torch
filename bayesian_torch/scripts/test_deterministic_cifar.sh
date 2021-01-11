#!/bin/bash

model=resnet20
mode='test'
batch_size=1000

python examples/main_deterministic_cifar.py --arch=$model --mode=$mode --batch-size=$batch_size
