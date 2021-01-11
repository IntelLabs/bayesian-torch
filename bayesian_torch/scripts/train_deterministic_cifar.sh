#!/bin/bash

model=resnet20
mode='train'
batch_size=512

python examples/main_deterministic_cifar.py --arch=$model --mode=$mode --batch-size=$batch_size
