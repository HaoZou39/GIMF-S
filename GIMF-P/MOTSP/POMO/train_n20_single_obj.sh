#!/bin/bash
##########################################################################################
# Training Script for MOTSP N20 Single Objective
# 
# Configuration Summary:
# - Problem size: 20 nodes
# - Single objective optimization (roadnet distance)
# - Basemap enabled with zscore normalization
# - Point representation: white_on_black + 3x3 dilation
# - Edge-aware auxiliary losses enabled (supervision + ranking)
# - Edge bias disabled (不影响路径选择过程)
# - No pretrain stage (直接开始训练)
#
# Usage:
#   ./train_n20_single_obj.sh           # Use default GPU 0
#   ./train_n20_single_obj.sh 1         # Use GPU 1
#   ./train_n20_single_obj.sh 0 debug   # Debug mode on GPU 0
##########################################################################################

#=============================================================================
# Configurable Parameters (修改这里的参数即可)
#=============================================================================

# Basic settings
GPU=${1:-0}                     # GPU device number

# Training settings
EPOCHS=200                      # Number of training epochs
BATCH_SIZE=64                   # Training batch size
LEARNING_RATE=1e-4              # Learning rate
RANDOM_SEED=1234                # Random seed for reproducibility
VALIDATION_INTERVAL=10          # Validate every N epochs (0 to disable)

# Environment settings
USE_BASEMAP=1                   # Use basemap as additional channel (1=True, 0=False)
POINT_STYLE="white_on_black"    # Point representation: "white_on_black" or "black_on_white"
POINT_DILATION="3x3"            # Point dilation: "3x3" or "1x1"
BASEMAP_NORMALIZE="zscore"      # Basemap normalization: "none" or "zscore"
BASEMAP_NORM_CLIP=3.0           # Clip value after zscore (0 to disable)
USE_DISTANCE_MATRIX=1           # Use roadnet distance matrix (1=True, 0=False)

# Model settings
USE_EDGE_HEAD=1                 # Enable edge prediction head (1=True, 0=False)
USE_EDGE_BIAS=0                 # Enable edge bias in decoder (0=不影响路径选择)

# Auxiliary loss settings
EDGE_PRETRAIN_ENABLE=0          # Enable pretrain stage (0=无预训练)
EDGE_PRETRAIN_EPOCHS=5          # Pretrain epochs (only used if EDGE_PRETRAIN_ENABLE=1)
EDGE_SUP_ENABLE=1               # Enable edge supervised loss
EDGE_SUP_WEIGHT=1.0             # Edge supervised loss weight
EDGE_RANK_ENABLE=1              # Enable edge ranking loss
EDGE_RANK_WEIGHT=0.1            # Edge ranking loss weight

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
echo "MOTSP N20 Single Objective Training"
echo "=============================================="
echo "GPU Device: $GPU"
echo ""
echo "Environment Configuration:"
echo "  - problem_size: 20"
echo "  - pomo_size: 20"
echo "  - num_objectives: 1"
echo "  - use_basemap: $USE_BASEMAP"
echo "  - point_style: $POINT_STYLE"
echo "  - point_dilation: $POINT_DILATION"
echo "  - basemap_normalize: $BASEMAP_NORMALIZE"
echo "  - basemap_norm_clip: $BASEMAP_NORM_CLIP"
echo "  - use_distance_matrix: $USE_DISTANCE_MATRIX"
echo ""
echo "Model Configuration:"
echo "  - embedding_dim: 128"
echo "  - encoder_layer_num: 6"
echo "  - in_channels: $([ $USE_BASEMAP -eq 1 ] && echo '2' || echo '1')"
echo "  - pixel_density: 56 (img_size=256)"
echo "  - use_edge_head: $USE_EDGE_HEAD"
echo "  - use_edge_bias: $USE_EDGE_BIAS"
echo ""
echo "Training Configuration:"
echo "  - epochs: $EPOCHS"
echo "  - train_batch_size: $BATCH_SIZE"
echo "  - learning_rate: $LEARNING_RATE"
echo "  - random_seed: $RANDOM_SEED"
echo "  - validation_interval: $VALIDATION_INTERVAL"
echo ""
echo "Edge-aware Auxiliary Losses:"
echo "  - edge_pretrain: enable=$EDGE_PRETRAIN_ENABLE, epochs=$EDGE_PRETRAIN_EPOCHS"
echo "  - edge_supervised: enable=$EDGE_SUP_ENABLE, weight=$EDGE_SUP_WEIGHT"
echo "  - edge_ranking: enable=$EDGE_RANK_ENABLE, weight=$EDGE_RANK_WEIGHT"
echo "=============================================="
echo ""

#=============================================================================
# Run training
#=============================================================================
cd "$(dirname "$0")"

echo "Starting training..."
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
    echo "Training completed successfully!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Training failed with exit code $?"
    echo "=============================================="
    exit 1
fi
