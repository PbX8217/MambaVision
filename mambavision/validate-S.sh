#!/bin/bash

DATA_PATH="/mnt/data/imagenet-1K/val"
BS=512
checkpoint='/mnt/data/mambavision/mamba_out_vision_S-224/model_best.pth.tar'


python validate.py --model mamba_vision_S --checkpoint=$checkpoint --data-dir=$DATA_PATH --batch-size $BS --input-size 3 224 224 --amp
