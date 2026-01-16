#!/bin/bash
# Test three models on ALL datasets (16 locations)
# 
# Training locations (12): 30.175448_*, 30.229377_*, 30.283276_* (each with 4 longitudes)
# Test locations (4 unseen): 30.337145_* (4 longitudes)

# Model checkpoints
# Baseline 1: no_basemap model
CKPT_NO_BASEMAP="./result/20260116_013519_train__tsp_n20_single_obj_12loc_newdata_no_basemap_whiteB1px_roadnet/checkpoint_motsp-50.pt"

# Baseline 2: basemap + whiteB1x1 model
CKPT_BASEMAP_1X1="./result/20260116_014643_train__tsp_n20_single_obj_12loc_newdata_ch2_whiteB1x1_roadnet/checkpoint_motsp-50.pt"

# New model: basemap + blackW3x3 + aux_pretrain (sup1.0_rank0.1_bias_off)
CKPT_BASEMAP_3X3="./result/20260116_174455_train__tsp_n20_single_obj_12loc_newdata_ch2_blackW3x3_roadnet_aux_pretrain5ep_sup1.0_rank0.1_bias_off/checkpoint_motsp-50.pt"

# Check if model files exist
if [ ! -f "$CKPT_NO_BASEMAP" ]; then
    echo "Error: Model 1 (no_basemap) not found: $CKPT_NO_BASEMAP"
    exit 1
fi

if [ ! -f "$CKPT_BASEMAP_1X1" ]; then
    echo "Error: Model 2 (basemap_whiteB1x1) not found: $CKPT_BASEMAP_1X1"
    exit 1
fi

if [ ! -f "$CKPT_BASEMAP_3X3" ]; then
    echo "Error: Model 3 (basemap_blackW3x3_aux) not found: $CKPT_BASEMAP_3X3"
    exit 1
fi

echo "=============================================================="
echo "Testing three models on ALL 16 datasets"
echo "=============================================================="
echo "Model 1 (no_basemap): $CKPT_NO_BASEMAP"
echo "Model 2 (basemap_whiteB1x1): $CKPT_BASEMAP_1X1"
echo "Model 3 (basemap_blackW3x3_aux): $CKPT_BASEMAP_3X3"
echo ""
echo "Datasets:"
echo "  Training (12 locations): 30.175448_*, 30.229377_*, 30.283276_*"
echo "  Test (4 unseen): 30.337145_*"
echo ""

# Run the test
python test_model.py --test_all \
    --ckpt_no_basemap "$CKPT_NO_BASEMAP" \
    --ckpt_basemap_black_1x1 "$CKPT_BASEMAP_1X1" \
    --ckpt_basemap_white_3x3 "$CKPT_BASEMAP_3X3" \
    --gpu 0 \
    --batch_size 64

echo ""
echo "Test completed!"
