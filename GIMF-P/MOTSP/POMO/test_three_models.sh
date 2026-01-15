#!/bin/bash
# 测试三个模型的脚本示例
# 
# 使用方法：
# 1. 修改下面的checkpoint路径为你的实际模型路径
# 2. 运行: bash test_three_models.sh

# 设置checkpoint路径（根据find_checkpoints.py的结果更新）
# 使用相同数据集训练的模型（杭州_上海_柏林_鹤岗），都是epoch 50

# 原始不加basemap的模型
CKPT_NO_BASEMAP="./result/20260112_080417_train__tsp_n20_single_obj_杭州_上海_柏林_鹤岗_no_basemap_roadnet/checkpoint_motsp-50.pt"

# 加了basemap但没有放大点的模型（ch2_roadnet）
CKPT_BASEMAP_1X1="./result/20260112_080702_train__tsp_n20_single_obj_杭州_上海_柏林_鹤岗_ch2_roadnet/checkpoint_motsp-50.pt"

# 加了basemap并且放大点的模型（ch2_blackW3x3_roadnet）
CKPT_BASEMAP_3X3="./result/20260112_084753_train__tsp_n20_single_obj_杭州_上海_柏林_鹤岗_ch2_blackW3x3_roadnet/checkpoint_motsp-50.pt"

# 运行测试
# 注意：如果只有两个模型，可以分别测试，或者先训练第三个模型

# 检查三个模型文件是否存在
if [ ! -f "$CKPT_NO_BASEMAP" ]; then
    echo "错误: 找不到模型1 (no_basemap): $CKPT_NO_BASEMAP"
    exit 1
fi

if [ ! -f "$CKPT_BASEMAP_1X1" ]; then
    echo "错误: 找不到模型2 (basemap_blackW1x1): $CKPT_BASEMAP_1X1"
    exit 1
fi

if [ ! -f "$CKPT_BASEMAP_3X3" ]; then
    echo "错误: 找不到模型3 (basemap_whiteB3x3): $CKPT_BASEMAP_3X3"
    exit 1
fi

# 三个模型都齐全，使用 --test_all 模式一起测试
echo "找到三个模型，开始测试..."
echo "模型1 (no_basemap): $CKPT_NO_BASEMAP"
echo "模型2 (basemap_blackW1x1): $CKPT_BASEMAP_1X1"
echo "模型3 (basemap_whiteB3x3): $CKPT_BASEMAP_3X3"
echo ""

python test_model.py --test_all \
    --ckpt_no_basemap "$CKPT_NO_BASEMAP" \
    --ckpt_basemap_black_1x1 "$CKPT_BASEMAP_1X1" \
    --ckpt_basemap_white_3x3 "$CKPT_BASEMAP_3X3" \
    --gpu 0 \
    --batch_size 64

# 如果只想测试特定数据集，可以添加 --datasets 参数
# 例如：--datasets 杭州 上海
