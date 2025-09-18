#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本功能：将label_data0615_0915文件夹中每个CSV文件的is_outlier列数据
替换到label_data文件夹中同名CSV文件的is_outlier列
"""

import os
import pandas as pd
from pathlib import Path

def update_outlier_labels():
    """
    更新is_outlier标签数据
    """
    # 定义源文件夹和目标文件夹路径
    source_dir = Path("label_data0615_0915")
    target_dir = Path("label_data")
    
    # 检查文件夹是否存在
    if not source_dir.exists():
        print(f"错误：源文件夹 {source_dir} 不存在")
        return
    
    if not target_dir.exists():
        print(f"错误：目标文件夹 {target_dir} 不存在")
        return
    
    # 获取源文件夹中的所有CSV文件
    source_files = list(source_dir.glob("*.csv"))
    
    if not source_files:
        print(f"警告：源文件夹 {source_dir} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(source_files)} 个源文件")
    
    updated_count = 0
    error_count = 0
    
    # 遍历每个源文件
    for source_file in source_files:
        try:
            # 构造目标文件路径
            target_file = target_dir / source_file.name
            
            # 检查目标文件是否存在
            if not target_file.exists():
                print(f"警告：目标文件 {target_file} 不存在，跳过")
                continue
            
            # 读取源文件和目标文件
            print(f"处理文件：{source_file.name}")
            
            source_df = pd.read_csv(source_file)
            target_df = pd.read_csv(target_file)
            
            # 检查必要的列是否存在
            if 'is_outlier' not in source_df.columns:
                print(f"错误：源文件 {source_file.name} 中没有 is_outlier 列")
                error_count += 1
                continue
                
            if 'is_outlier' not in target_df.columns:
                print(f"错误：目标文件 {target_file.name} 中没有 is_outlier 列")
                error_count += 1
                continue
            
            # 检查msgTime列是否存在，用于数据对齐
            if 'msgTime' not in source_df.columns or 'msgTime' not in target_df.columns:
                print(f"错误：文件 {source_file.name} 缺少 msgTime 列，无法进行数据对齐")
                error_count += 1
                continue
            
            # 将msgTime转换为datetime格式进行匹配
            source_df['msgTime'] = pd.to_datetime(source_df['msgTime'])
            target_df['msgTime'] = pd.to_datetime(target_df['msgTime'])
            
            # 创建源数据的时间-标签映射
            source_outlier_map = dict(zip(source_df['msgTime'], source_df['is_outlier']))
            
            # 更新目标文件的is_outlier列
            original_outliers = target_df['is_outlier'].copy()
            
            # 对于目标文件中的每个时间点，如果在源文件中存在对应时间，则更新is_outlier值
            for idx, row in target_df.iterrows():
                if row['msgTime'] in source_outlier_map:
                    target_df.at[idx, 'is_outlier'] = source_outlier_map[row['msgTime']]
            
            # 统计更新的记录数
            updated_records = (original_outliers != target_df['is_outlier']).sum()
            
            # 保存更新后的文件
            target_df.to_csv(target_file, index=False)
            
            print(f"  ✓ 成功更新 {updated_records} 条记录")
            updated_count += 1
            
        except Exception as e:
            print(f"错误：处理文件 {source_file.name} 时发生异常：{str(e)}")
            error_count += 1
    
    # 输出总结
    print(f"\n=== 处理完成 ===")
    print(f"成功更新文件数：{updated_count}")
    print(f"处理失败文件数：{error_count}")
    print(f"总文件数：{len(source_files)}")

if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("开始更新is_outlier标签数据...")
    update_outlier_labels()
    print("脚本执行完成！")