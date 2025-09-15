import pandas as pd
import os
from datetime import datetime

def convert_smooth_to_label_format():
    """
    将clean_results_smooth目录中的device开头CSV文件转换为label_data格式
    """
    
    # 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    smooth_dir = os.path.join(script_dir, 'clean_results_smooth')
    label_dir = os.path.join(script_dir, 'label_data')
    
    # 确保输出目录存在
    os.makedirs(label_dir, exist_ok=True)
    
    # 获取所有device开头的CSV文件
    csv_files = [f for f in os.listdir(smooth_dir) 
                 if f.startswith('device_') and f.endswith('.csv')]
    
    print(f"找到 {len(csv_files)} 个设备CSV文件")
    
    for csv_file in csv_files:
        try:
            # 读取源文件
            source_path = os.path.join(smooth_dir, csv_file)
            df = pd.read_csv(source_path)
            
            # 只保留需要的4个字段
            required_columns = ['msgTime', 'liquidLevelValue', 'liquidLevel_clean', 'is_outlier']
            df_filtered = df[required_columns].copy()
            
            # 转换时间格式
            df_filtered['msgTime'] = pd.to_datetime(df_filtered['msgTime'])
            df_filtered['msgTime'] = df_filtered['msgTime'].dt.strftime('%Y/%m/%d %H:%M')
            
            # 转换is_outlier为整数格式
            df_filtered['is_outlier'] = df_filtered['is_outlier'].astype(int)
            
            # 保存到label_data目录
            output_path = os.path.join(label_dir, csv_file)
            df_filtered.to_csv(output_path, index=False)
            
            print(f"✓ 已转换: {csv_file} ({len(df_filtered)} 行数据)")
            
        except Exception as e:
            print(f"✗ 转换失败: {csv_file} - 错误: {str(e)}")
    
    print("\n转换完成！")

def verify_conversion():
    """
    验证转换结果
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_dir = os.path.join(script_dir, 'label_data')
    
    # 检查转换后的文件
    converted_files = [f for f in os.listdir(label_dir) 
                      if f.startswith('device_') and f.endswith('.csv')]
    
    print(f"\n验证结果：")
    print(f"转换后的文件数量: {len(converted_files)}")
    
    # 随机检查一个文件的格式
    if converted_files:
        sample_file = converted_files[0]
        sample_path = os.path.join(label_dir, sample_file)
        df_sample = pd.read_csv(sample_path)
        
        print(f"\n示例文件: {sample_file}")
        print(f"字段: {list(df_sample.columns)}")
        print(f"数据行数: {len(df_sample)}")
        print(f"前3行数据:")
        print(df_sample.head(3).to_string(index=False))

if __name__ == "__main__":
    print("开始转换clean_results_smooth到label_data格式...")
    convert_smooth_to_label_format()
    verify_conversion()