#!/bin/bash

DATA_PATH="/mnt/data/imagenet-1K/val"
BS=1024
checkpoint='/mnt/data/mambavision/mamba_out_vision_T-224/model_best.pth.tar'


python validate.py --model mamba_vision_T --checkpoint=$checkpoint --data-dir=$DATA_PATH --batch-size $BS --input-size 3 224 224 --amp
