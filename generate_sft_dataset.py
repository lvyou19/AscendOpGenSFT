#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AscendC SFT Dataset Generator
从工程路径下的算子目录生成Alpaca格式的训练数据集
"""

import os
import json
from pathlib import Path

# 配置路径
root_dir = "/home/l00868164/ops_dataset"
output_dir = "/home/l00868164/ops_dataset/output"
os.makedirs(output_dir, exist_ok=True)


def find_valid_op_dirs(root):
    """查找所有包含op_host和op_kernel的有效算子目录"""
    valid_ops = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过tests目录
        if 'tests' in dirpath.split(os.sep):
            continue
            
        op_host_path = os.path.join(dirpath, 'op_host')
        op_kernel_path = os.path.join(dirpath, 'op_kernel')
        
        if os.path.isdir(op_host_path) and os.path.isdir(op_kernel_path):
            # 从目录名获取算子名称
            op_name = os.path.basename(dirpath)
            valid_ops.append({
                'op_name': op_name,
                'op_path': dirpath,
                'op_host': op_host_path,
                'op_kernel': op_kernel_path
            })
    
    return valid_ops


def collect_source_files(op_host_path, op_kernel_path):
    """收集op_host和op_kernel中的.cpp和.h文件（排除config和op_api目录）"""
    files_content = []
    
    for base_path in [op_host_path, op_kernel_path]:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            # 跳过config和op_api目录
            if item in ['config', 'op_api']:
                continue
            # 跳过子目录
            if os.path.isdir(item_path):
                continue
            # 只包含.cpp和.h文件
            if item.endswith(('.cpp', '.h')):
                try:
                    with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        files_content.append({
                            'filename': item,
                            'filepath': item_path,
                            'content': content
                        })
                except Exception as e:
                    print(f"  读取文件出错 {item_path}: {e}")
    
    return files_content


def create_alpaca_entry(op_name, files_content):
    """创建Alpaca格式的训练数据条目"""
    # 构建output内容
    output_parts = []
    for file_info in files_content:
        output_parts.append(
            f"```cpp\n// File: {file_info['filename']}\n{file_info['content']}\n```"
        )
    
    output = "\n============================================================================================\n".join(output_parts)
    
    return {
        "instruction": f"请生成Ascend C高性能kernel算子 {op_name} 的完整实现代码",
        "input": f"帮我生成Ascend C高性能kernel算子，算子名称为{op_name}",
        "output": output,
        "system": "你是基于华为昇腾处理器AscendC编程专家，擅长生成高性能的算子实现代码。",
        "history": []
    }


def main():
    """主函数：生成SFT训练数据集"""
    print("=" * 60)
    print("AscendC SFT Dataset Generator")
    print("=" * 60)
    
    # 1. 查找所有有效算子目录
    print("\n[1/4] 扫描有效算子目录...")
    valid_ops = find_valid_op_dirs(root_dir)
    print(f"找到 {len(valid_ops)} 个有效算子目录")
    
    # 2. 处理每个算子
    print("\n[2/4] 处理算子并生成训练数据...")
    dataset = []
    skipped = []
    
    for idx, op_info in enumerate(valid_ops):
        op_name = op_info['op_name']
        print(f"  处理 [{idx+1}/{len(valid_ops)}]: {op_name}")
        
        # 收集源文件
        files_content = collect_source_files(op_info['op_host'], op_info['op_kernel'])
        
        if not files_content:
            print(f"    ⚠️  警告: 未找到源文件，跳过")
            skipped.append(op_name)
            continue
        
        print(f"    ✓ 收集到 {len(files_content)} 个源文件")
        
        # 创建Alpaca条目
        entry = create_alpaca_entry(op_name, files_content)
        dataset.append(entry)
        
        # 保存单个JSON文件
        output_filename = f"{op_name}_ImplTag_{idx}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        print(f"    ✓ 已保存: {output_filename}")
    
    # 3. 保存合并数据集
    print("\n[3/4] 保存合并数据集...")
    combined_output_path = os.path.join(output_dir, "ascendc_sft_dataset.json")
    with open(combined_output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # 4. 输出统计信息
    print("\n[4/4] 生成统计报告")
    print("=" * 60)
    print(f"总算子数: {len(valid_ops)}")
    print(f"成功处理: {len(dataset)}")
    print(f"跳过算子: {len(skipped)}")
    if skipped:
        print(f"跳过的算子: {', '.join(skipped)}")
    print(f"\n输出目录: {output_dir}")
    print(f"合并数据集: {combined_output_path}")
    print(f"数据集大小: {os.path.getsize(combined_output_path) / 1024 / 1024:.2f} MB")
    print("=" * 60)
    print("数据集生成完成！")


if __name__ == "__main__":
    main()
