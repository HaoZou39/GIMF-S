#!/usr/bin/env python3
"""
查找可用的模型checkpoint文件

用法：
    python find_checkpoints.py
    python find_checkpoints.py --pattern "no_basemap"
"""

import os
import argparse
import glob

def find_checkpoints(pattern=None):
    """查找result目录下的所有checkpoint文件"""
    result_dir = "./result"
    
    if not os.path.exists(result_dir):
        print(f"结果目录不存在: {result_dir}")
        return
    
    # 查找所有checkpoint文件
    checkpoint_pattern = os.path.join(result_dir, "**", "checkpoint_motsp-*.pt")
    checkpoints = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoints:
        print("未找到任何checkpoint文件")
        return
    
    # 按配置分类
    configs = {
        'no_basemap': [],
        'basemap_blackW1x1': [],
        'basemap_whiteB3x3': [],
    }
    
    for ckpt in sorted(checkpoints):
        ckpt_lower = ckpt.lower()
        
        # 判断配置类型
        if 'no_basemap' in ckpt_lower:
            configs['no_basemap'].append(ckpt)
        elif 'blackw3x3' in ckpt_lower or ('ch2' in ckpt_lower and '3x3' in ckpt_lower):
            # ch2_blackW3x3_roadnet - 加了basemap且放大点
            configs['basemap_whiteB3x3'].append(ckpt)
        elif 'ch2' in ckpt_lower and 'blackw3x3' not in ckpt_lower:
            # ch2_roadnet - 加了basemap但未放大点（默认1x1）
            configs['basemap_blackW1x1'].append(ckpt)
    
    # 打印结果
    print("=" * 80)
    print("找到的Checkpoint文件：")
    print("=" * 80)
    
    for config_name, ckpts in configs.items():
        if ckpts:
            print(f"\n{config_name}:")
            for ckpt in ckpts:
                # 提取epoch信息
                epoch = os.path.basename(ckpt).replace('checkpoint_motsp-', '').replace('.pt', '')
                print(f"  Epoch {epoch}: {ckpt}")
        else:
            print(f"\n{config_name}: (未找到)")
    
    # 生成测试命令
    print("\n" + "=" * 80)
    print("建议的测试命令：")
    print("=" * 80)
    
    if all(ckpts for ckpts in configs.values()):
        print("\npython test_model.py --test_all \\")
        if configs['no_basemap']:
            print(f"    --ckpt_no_basemap {configs['no_basemap'][-1]} \\")
        if configs['basemap_blackW1x1']:
            print(f"    --ckpt_basemap_black_1x1 {configs['basemap_blackW1x1'][-1]} \\")
        if configs['basemap_whiteB3x3']:
            print(f"    --ckpt_basemap_white_3x3 {configs['basemap_whiteB3x3'][-1]} \\")
        print("    --gpu 0 --batch_size 64")
    else:
        print("\n注意：需要三个配置的checkpoint才能使用 --test_all 模式")
        print("或者可以单独测试每个模型：")
        print("\n# 测试单个模型")
        print("python test_model.py --checkpoint <checkpoint路径> --config <config_name> --gpu 0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查找可用的模型checkpoint')
    parser.add_argument('--pattern', type=str, default=None,
                        help='过滤模式（例如：no_basemap）')
    args = parser.parse_args()
    
    find_checkpoints(args.pattern)
