#!/bin/bash
##########################################################################################
# Training Script for MOTSP N20 Single Objective - BASELINE
# 
# Baseline Configuration (无任何增强技术):
# - Problem size: 20 nodes
# - Single objective optimization (roadnet distance)
# - NO basemap (纯坐标点图像)
# - Point representation: black_on_white + 1x1 (黑点白底，无膨胀)
# - NO edge-aware auxiliary losses (无辅助损失)
# - NO edge head / edge bias (无Edge预测头)
#
# Usage:
#   ./train_n20_single_obj_baseline.sh           # Use default GPU 0
#   ./train_n20_single_obj_baseline.sh 1         # Use GPU 1
#   ./train_n20_single_obj_baseline.sh 0 debug   # Debug mode on GPU 0
##########################################################################################

#=============================================================================
# Configurable Parameters - BASELINE (修改这里的参数即可)
#=============================================================================

# Basic settings
GPU=${1:-0}                     # GPU device number

# Training settings
EPOCHS=200                      # Number of training epochs
BATCH_SIZE=64                   # Training batch size
LEARNING_RATE=1e-4              # Learning rate
RANDOM_SEED=1234                # Random seed for reproducibility
VALIDATION_INTERVAL=10          # Validate every N epochs (0 to disable)

# Environment settings - BASELINE
USE_BASEMAP=0                   # NO basemap (纯坐标点图像)
POINT_STYLE="black_on_white"    # 黑点白底 (原始风格)
POINT_DILATION="1x1"            # 无膨胀，单像素
BASEMAP_NORMALIZE="none"        # 无归一化
BASEMAP_NORM_CLIP=0             # 无clip
USE_DISTANCE_MATRIX=1           # Use roadnet distance matrix (1=True, 0=False)

# Model settings - BASELINE (无Edge模块)
USE_EDGE_HEAD=0                 # NO edge prediction head
USE_EDGE_BIAS=0                 # NO edge bias

# Auxiliary loss settings - BASELINE (全部禁用)
EDGE_PRETRAIN_ENABLE=0          # NO pretrain stage
EDGE_PRETRAIN_EPOCHS=0          # N/A
EDGE_SUP_ENABLE=0               # NO edge supervised loss
EDGE_SUP_WEIGHT=0.0             # N/A
EDGE_RANK_ENABLE=0              # NO edge ranking loss
EDGE_RANK_WEIGHT=0.0            # N/A

#=============================================================================
# Debug mode check
#=============================================================================
DEBUG_FLAG=""
if [ "$2" == "debug" ]; then
    DEBUG_FLAG="--debug"
    echo "=============================================="
    echo "Running in DEBUG mode"
    echo "=============================================="
fi

#=============================================================================
# Display configuration
#=============================================================================
echo "=============================================="
echo "MOTSP N20 Single Objective Training - BASELINE"
echo "=============================================="
echo "GPU Device: $GPU"
echo ""
echo "Environment Configuration (BASELINE):"
echo "  - problem_size: 20"
echo "  - pomo_size: 20"
echo "  - num_objectives: 1"
echo "  - use_basemap: $USE_BASEMAP (无basemap)"
echo "  - point_style: $POINT_STYLE (黑点白底)"
echo "  - point_dilation: $POINT_DILATION (无膨胀)"
echo "  - basemap_normalize: $BASEMAP_NORMALIZE"
echo "  - use_distance_matrix: $USE_DISTANCE_MATRIX"
echo ""
echo "Model Configuration (BASELINE):"
echo "  - embedding_dim: 128"
echo "  - encoder_layer_num: 6"
echo "  - in_channels: 1 (仅坐标点)"
echo "  - pixel_density: 56 (img_size=256)"
echo "  - use_edge_head: $USE_EDGE_HEAD (无Edge预测头)"
echo "  - use_edge_bias: $USE_EDGE_BIAS"
echo ""
echo "Training Configuration:"
echo "  - epochs: $EPOCHS"
echo "  - train_batch_size: $BATCH_SIZE"
echo "  - learning_rate: $LEARNING_RATE"
echo "  - random_seed: $RANDOM_SEED"
echo "  - validation_interval: $VALIDATION_INTERVAL"
echo ""
echo "Edge-aware Auxiliary Losses (BASELINE - 全部禁用):"
echo "  - edge_pretrain: enable=$EDGE_PRETRAIN_ENABLE"
echo "  - edge_supervised: enable=$EDGE_SUP_ENABLE"
echo "  - edge_ranking: enable=$EDGE_RANK_ENABLE"
echo "=============================================="
echo ""

#=============================================================================
# Run training
#=============================================================================
cd "$(dirname "$0")"

echo "Starting BASELINE training..."
CMD="python train_motsp_n20_single_obj.py \
    --gpu $GPU \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --seed $RANDOM_SEED \
    --validation_interval $VALIDATION_INTERVAL \
    --use_basemap $USE_BASEMAP \
    --point_style $POINT_STYLE \
    --point_dilation $POINT_DILATION \
    --basemap_normalize $BASEMAP_NORMALIZE \
    --basemap_norm_clip $BASEMAP_NORM_CLIP \
    --use_distance_matrix $USE_DISTANCE_MATRIX \
    --use_edge_head $USE_EDGE_HEAD \
    --use_edge_bias $USE_EDGE_BIAS \
    --edge_pretrain_enable $EDGE_PRETRAIN_ENABLE \
    --edge_pretrain_epochs $EDGE_PRETRAIN_EPOCHS \
    --edge_sup_enable $EDGE_SUP_ENABLE \
    --edge_sup_weight $EDGE_SUP_WEIGHT \
    --edge_rank_enable $EDGE_RANK_ENABLE \
    --edge_rank_weight $EDGE_RANK_WEIGHT \
    $DEBUG_FLAG"

echo "Command: $CMD"
echo ""

eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "BASELINE Training completed successfully!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "BASELINE Training failed with exit code $?"
    echo "=============================================="
    exit 1
fi
