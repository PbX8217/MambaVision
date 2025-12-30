#!/bin/bash

# Single-node multi-GPU training with torchrun (no Slurm)
# Finetune Cascade Mask R-CNN + MambaVision-T backbone on COCO
MODEL=mamba_vision_T

RUN_NAME=mamba_vision_t_object_detection
TOTAL_BATCH_SIZE=16

# ---------------- User params ----------------
NUM_GPUS=${NUM_GPUS:-1}  # Number of GPUs to use
BATCH_SIZE_PER_GPU=$((TOTAL_BATCH_SIZE / NUM_GPUS))
CONFIG="./configs/mamba_vision/mask_rcnn_mamba_vision_t_1x_coco.py"
PRETRAIN="/mnt/data/mambavision/${MODEL}-224/model_best.pth.tar"  # Path to pretrained weights
WORK_DIR=./work_dirs/${RUN_NAME}
MASTER_PORT=28001
LOG_DIR=${LOG_DIR:-./log_dirs/${RUN_NAME}}

# ---------------- Launch ----------------
export RUN_NAME TOTAL_BATCH_SIZE NUM_GPUS BATCH_SIZE_PER_GPU LOG_DIR

mkdir -p "${WORK_DIR}" "${LOG_DIR}"

if [ $NUM_GPUS -eq 1 ]; then
  echo "${RUN_NAME} with single GPU"
  python tools/train.py ${CONFIG} --work-dir ${WORK_DIR} --amp --resume
else
  echo "${RUN_NAME} with ${NUM_GPUS} GPUs"
  torchrun --nproc_per_node=${NUM_GPUS} --rdzv_endpoint=localhost:${MASTER_PORT} \
    tools/train.py ${CONFIG} --work-dir ${WORK_DIR} --amp --launcher pytorch --resume
fi