#!/bin/bash

# ---------------- 配置区域 ----------------
# 模型型号: mamba_vision_T, mamba_vision_S, mamba_vision_B, mamba_vision_L
MODEL=mamba_vision_T

# 数据集路径 (请确保指向转换后的文件夹结构)
DATA_PATH_TRAIN="/mnt/data/imagenet-1K/train"
DATA_PATH_VAL="/mnt/data/imagenet-1K/val"

# 实验名称
EXP=mamba_vision_t_1k_scratch

# 端口设置 (防止多任务冲突，同时运行多个训练时请修改此值)
MASTER_PORT=29002

# 从checkpoint恢复训练 (如果需要，从之前的实验中恢复)
# 注意: 必须指向具体的 .pth.tar 文件，不能是文件夹！
# 推荐使用 last.pth.tar 来恢复中断的训练
RESUME_CHECKPOINT="/mnt/data/mambavision/mamba_vision_T-224/last.pth.tar"

# 硬件配置
NUM_GPUS=${NUM_GPUS:-4}  # Number of GPUs to use (default: 4)
# 单卡 Batch Size (根据显存调整: 128, 64, 32...)
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-512}

# --- 原始论文预训练设置 (Pre-training) ---
# 总 Batch Size = 4096
# Optimizer = LAMB
# LR = 4e-3
# Epochs = 300
TOTAL_BATCH_SIZE=4096
OPT="lamb"
LR=4e-3
EPOCHS=200
WARMUP_EPOCHS=20
WD=0.05
DROP_PATH=0.1

# 自动计算梯度累积步数
# Accumulation Steps = Total Batch Size / (Num GPUs * Batch Size Per GPU)
GLOBAL_BATCH_SIZE_CURRENT=$((NUM_GPUS * BATCH_SIZE_PER_GPU))
ACCUM_STEPS=$((TOTAL_BATCH_SIZE / GLOBAL_BATCH_SIZE_CURRENT))

echo "------------------------------------------------"
echo "Model: $MODEL"
echo "Total Batch Size Target: $TOTAL_BATCH_SIZE"
echo "GPUs: $NUM_GPUS"
echo "Batch Size Per GPU: $BATCH_SIZE_PER_GPU"
echo "Calculated Accumulation Steps: $ACCUM_STEPS"
echo "Effective Batch Size: $((GLOBAL_BATCH_SIZE_CURRENT * ACCUM_STEPS))"
echo "Optimizer: $OPT"
echo "LR: $LR"
echo "------------------------------------------------"

# 运行训练
RESUME_ARGS=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ARGS="--resume $RESUME_CHECKPOINT"
fi

torchrun --nproc_per_node=$NUM_GPUS --rdzv_endpoint=localhost:$MASTER_PORT train.py \
  --model $MODEL \
  --data_dir /mnt/data/imagenet-1K \
  --lmdb_dataset \
  --train-split train --val-split val \
  --input-size 3 224 224 --crop-pct=0.875 \
  --batch-size $BATCH_SIZE_PER_GPU \
  --epochs $EPOCHS \
  --warmup-epochs $WARMUP_EPOCHS \
  --opt $OPT \
  --lr $LR \
  --weight-decay $WD \
  --drop-path $DROP_PATH \
  --amp \
  --pin-mem \
  --workers 4 \
  --tag $EXP \
  --grad-accum-steps $ACCUM_STEPS \
  $RESUME_ARGS

# --- 微调设置 (Fine-tuning) 参考 ---
# 如果需要微调 (例如更高分辨率或后期优化)，通常使用:
# OPT="adamw"
# LR=1e-5 (或其他较小值)
# TOTAL_BATCH_SIZE=512 (或更小)
# 并在 train.py 中加载 --initial-checkpoint

