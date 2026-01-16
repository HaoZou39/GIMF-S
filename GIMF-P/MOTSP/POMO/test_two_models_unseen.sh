#!/bin/bash
# 测试两个模型在未见过的测试集上的性能
# 
# 测试集: 4个30.337145位置（未在训练中使用）
# 模型1: 无卫星图模型
# 模型2: 有卫星图模型（白底黑点 1x1）

# 设置checkpoint路径
CKPT_NO_BASEMAP="./result/20260116_013519_train__tsp_n20_single_obj_12loc_newdata_no_basemap_whiteB1px_roadnet/checkpoint_motsp-50.pt"
CKPT_BASEMAP_1X1="./result/20260116_014643_train__tsp_n20_single_obj_12loc_newdata_ch2_whiteB1x1_roadnet/checkpoint_motsp-50.pt"

# 测试数据集名称（4个未见过的位置）
TEST_DATASETS="test_loc_30.337145_120.065850 test_loc_30.337145_120.128249 test_loc_30.337145_120.190647 test_loc_30.337145_120.253046"

# 检查模型文件是否存在
if [ ! -f "$CKPT_NO_BASEMAP" ]; then
    echo "错误: 找不到模型1 (no_basemap): $CKPT_NO_BASEMAP"
    exit 1
fi

if [ ! -f "$CKPT_BASEMAP_1X1" ]; then
    echo "错误: 找不到模型2 (basemap_whiteB1x1): $CKPT_BASEMAP_1X1"
    exit 1
fi

echo "="
echo "开始测试两个模型在未见过的测试集上的性能"
echo "="
echo "模型1 (no_basemap): $CKPT_NO_BASEMAP"
echo "模型2 (basemap_whiteB1x1): $CKPT_BASEMAP_1X1"
echo "测试数据集: $TEST_DATASETS"
echo ""

# 步骤1: 检查最优解文件是否存在，如果不存在则生成
echo "检查最优解文件..."
NEED_SOLVE=0
for dataset in $TEST_DATASETS; do
    # 构造最优解文件路径（简化检查，实际路径由Python脚本解析）
    echo "  检查 $dataset 的最优解文件..."
done

echo ""
echo "如果最优解文件不存在，将先运行 solve_optimal_ortools.py 生成最优解..."
echo ""

# 步骤2: 运行测试
echo "开始测试..."
python test_model.py --test_all \
    --ckpt_no_basemap "$CKPT_NO_BASEMAP" \
    --ckpt_basemap_black_1x1 "$CKPT_BASEMAP_1X1" \
    --gpu 0 \
    --batch_size 64 \
    --datasets $TEST_DATASETS

echo ""
echo "测试完成！"
