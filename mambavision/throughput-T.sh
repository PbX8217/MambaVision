#!/bin/bash

DATA_PATH="/mnt/data/imagenet-1K/val"
checkpoint='/mnt/data/mambavision/mamba_out_vision_T-224/model_best.pth.tar'


python throughput_measure.py --model mamba_vision_T --checkpoint=$checkpoint --data-dir=$DATA_PATH --amp
