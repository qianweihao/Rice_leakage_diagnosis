# -*- coding: utf-8 -*-
# 分析算法识别为漏水且与人工标注符合的事件的漏水速率范围
import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta

# ===================== 配置 =====================
LABEL_DATA_DIR = "label_data"
ALGO_RESULTS_FILE = "labeled_events_final_12h.csv"
PATTERN = "device_*.csv"
MERGE_GAP_HOURS = 2  # 人工标记合并间隔（小时）

# ===================== 数据加载 =====================
def load_manual_labels():
    """
    从 label_data 目录加载人工标记的排水事件点（is_outlier=1）
    返回: DataFrame[code, timestamp]
    """
    manual_drains = []

    for fp in glob.glob(os.path.join(LABEL_DATA_DIR, PATTERN)):
        code = os.path.basename(fp).replace("device_", "").replace(".csv", "")
        df = pd.read_csv(fp)

        if 'msgTime' not in df.columns or 'is_outlier' not in df.columns:
            print(f"[{code}] 缺少必要列（msgTime 或 is_outlier），跳过")
            continue

        df['msgTime'] = pd.to_datetime(df['msgTime'], errors='coerce')
        drain_events = df[df['is_outlier'] == 1].dropna(subset=['msgTime'])

        for _, row in drain_events.iterrows():
            manual_drains.append({
                'code': code,
                'timestamp': row['msgTime']
            })

    return pd.DataFrame(manual_drains)

def load_algo_results():
    """
    加载算法检测结果（CSV 需含列：code,start,end,label,level_start_mm,level_at_12h_mm,delta_12h_mm）
    返回: DataFrame[code, start, end, level_start_mm, level_at_12h_mm, delta_12h_mm] —— 仅保留 label=='drain'
    """
    if not os.path.exists(ALGO_RESULTS_FILE):
        print(f"算法结果文件 {ALGO_RESULTS_FILE} 不存在")
        return pd.DataFrame()

    df = pd.read_csv(ALGO_RESULTS_FILE)
    required_cols = {'code', 'start', 'end', 'label', 'level_start_mm', 'level_at_12h_mm', 'delta_12h_mm'}
    if not required_cols.issubset(df.columns):
        print(f"算法结果文件缺少必要列（需要 {required_cols}）")
        return pd.DataFrame()

    df = df[df['label'] == 'drain'].copy()
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    df = df.dropna(subset=['start', 'end', 'code'])

    return df[['code', 'start', 'end', 'level_start_mm', 'level_at_12h_mm', 'delta_12h_mm']]

# ===================== 人工事件合并 =====================
def get_manual_drain_periods(manual_df, merge_gap_hours=MERGE_GAP_HOURS):
    """
    将人工标记的单个时间点（is_outlier=1）合并为连续的"人工排水时间段"
    合并规则：相邻两个 1 的时间差 ≤ merge_gap_hours 小时视为同一段
    返回: DataFrame[code, start, end]
    """
    drain_periods = []

    for device in manual_df['code'].unique():
        device_data = manual_df[manual_df['code'] == device].sort_values('timestamp')

        if device_data.empty:
            continue

        periods = []
        start_time = None
        prev_time = None

        for _, row in device_data.iterrows():
            current_time = row['timestamp']

            if start_time is None:
                start_time = current_time
                prev_time = current_time
            else:
                # 是否连续（相差 ≤ merge_gap_hours 小时）
                if (current_time - prev_time) <= timedelta(hours=merge_gap_hours):
                    prev_time = current_time
                else:
                    periods.append((start_time, prev_time))
                    start_time = current_time
                    prev_time = current_time

        # 收尾
        if start_time is not None:
            periods.append((start_time, prev_time))

        for s, e in periods:
            drain_periods.append({'code': device, 'start': s, 'end': e})

    return pd.DataFrame(drain_periods)

# ===================== 时间重叠判断 =====================
def has_overlap(start1, end1, start2, end2):
    """
    判断两个时间段是否有重叠
    """
    return max(start1, start2) < min(end1, end2)

# ===================== 匹配事件并计算速率 =====================
def find_matched_events_with_rates(manual_periods_df, algo_df):
    """
    找到算法检测与人工标注匹配的事件，并计算漏水速率
    返回: DataFrame[code, algo_start, algo_end, manual_start, manual_end, 
                    level_start_mm, level_at_12h_mm, delta_12h_mm, 
                    duration_hours, drain_rate_mm_per_hour]
    """
    matched_events = []
    
    for device in set(manual_periods_df['code'].unique()) | set(algo_df['code'].unique()):
        m_dev = manual_periods_df[manual_periods_df['code'] == device]
        a_dev = algo_df[algo_df['code'] == device]
        
        # 对每个算法事件，寻找匹配的人工事件
        for _, ar in a_dev.iterrows():
            a_start, a_end = ar['start'], ar['end']
            
            for _, mr in m_dev.iterrows():
                m_start, m_end = mr['start'], mr['end']
                
                # 判断是否有时间重叠
                if has_overlap(a_start, a_end, m_start, m_end):
                    # 计算算法事件的持续时间（小时）
                    duration_hours = (a_end - a_start).total_seconds() / 3600
                    
                    # 计算漏水速率（mm/小时）
                    # 使用算法检测的液位变化和持续时间
                    drain_rate_mm_per_hour = abs(ar['delta_12h_mm']) / 12  # 12小时的变化除以12
                    
                    matched_events.append({
                        'code': device,
                        'algo_start': a_start,
                        'algo_end': a_end,
                        'manual_start': m_start,
                        'manual_end': m_end,
                        'level_start_mm': ar['level_start_mm'],
                        'level_at_12h_mm': ar['level_at_12h_mm'],
                        'delta_12h_mm': ar['delta_12h_mm'],
                        'duration_hours': duration_hours,
                        'drain_rate_mm_per_hour': drain_rate_mm_per_hour
                    })
                    break  # 找到匹配后跳出内层循环
    
    return pd.DataFrame(matched_events)

# ===================== 分析速率范围 =====================
def analyze_drain_rate_ranges(matched_events_df):
    """
    分析每个设备和总体的漏水速率范围
    """
    if matched_events_df.empty:
        print("没有找到匹配的漏水事件")
        return
    
    print("=== 算法识别与人工标注匹配的漏水事件速率分析 ===")
    print(f"总匹配事件数: {len(matched_events_df)}")
    print()
    
    # 按设备分析
    device_stats = []
    
    for device in sorted(matched_events_df['code'].unique()):
        device_data = matched_events_df[matched_events_df['code'] == device]
        rates = device_data['drain_rate_mm_per_hour']
        
        stats = {
            'device': device,
            'event_count': len(device_data),
            'min_rate': rates.min(),
            'max_rate': rates.max(),
            'mean_rate': rates.mean(),
            'median_rate': rates.median(),
            'std_rate': rates.std()
        }
        device_stats.append(stats)
        
        print(f"设备 {device}:")
        print(f"  匹配事件数: {stats['event_count']}")
        print(f"  速率范围: {stats['min_rate']:.3f} - {stats['max_rate']:.3f} mm/h")
        print(f"  平均速率: {stats['mean_rate']:.3f} mm/h")
        print(f"  中位数速率: {stats['median_rate']:.3f} mm/h")
        print(f"  标准差: {stats['std_rate']:.3f} mm/h")
        print()
    
    # 总体分析
    all_rates = matched_events_df['drain_rate_mm_per_hour']
    
    print("=== 总体漏水速率范围 ===")
    print(f"总体速率范围: {all_rates.min():.3f} - {all_rates.max():.3f} mm/h")
    print(f"总体平均速率: {all_rates.mean():.3f} mm/h")
    print(f"总体中位数速率: {all_rates.median():.3f} mm/h")
    print(f"总体标准差: {all_rates.std():.3f} mm/h")
    print()
    
    # 速率分布统计
    print("=== 速率分布统计 ===")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(all_rates, p)
        print(f"第{p}百分位数: {value:.3f} mm/h")
    print()
    
    # 速率区间分布
    print("=== 速率区间分布 ===")
    bins = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float('inf')]
    bin_labels = ['0-0.1', '0.1-0.2', '0.2-0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0', '>5.0']
    
    rate_counts = pd.cut(all_rates, bins=bins, labels=bin_labels, right=False).value_counts().sort_index()
    
    for interval, count in rate_counts.items():
        percentage = count / len(all_rates) * 100
        print(f"{interval} mm/h: {count}个事件 ({percentage:.1f}%)")
    
    # 保存详细结果
    device_stats_df = pd.DataFrame(device_stats)
    matched_events_df.to_csv('matched_drain_events_with_rates.csv', index=False, encoding='utf-8-sig')
    device_stats_df.to_csv('device_drain_rate_stats.csv', index=False, encoding='utf-8-sig')
    
    print("\n详细结果已保存到:")
    print("- matched_drain_events_with_rates.csv (所有匹配事件详情)")
    print("- device_drain_rate_stats.csv (各设备统计)")

# ===================== 主流程 =====================
def main():
    print("=== 漏水事件速率范围分析 ===")
    
    # 1) 加载人工标记
    print("\n1. 加载人工标记数据...")
    manual_df = load_manual_labels()
    print(f"   人工标记排水时间点总数: {len(manual_df)}")
    
    # 2) 加载算法结果
    print("\n2. 加载算法检测结果...")
    algo_df = load_algo_results()
    print(f"   算法检测排水事件总数: {len(algo_df)}")
    
    if manual_df.empty or algo_df.empty:
        print("\n数据不足，无法进行分析")
        return
    
    # 3) 合并人工事件
    print("\n3. 合并人工标记时间段...")
    manual_periods_df = get_manual_drain_periods(manual_df)
    print(f"   人工标记排水时间段总数: {len(manual_periods_df)}")
    
    # 4) 找到匹配事件并计算速率
    print("\n4. 匹配事件并计算漏水速率...")
    matched_events_df = find_matched_events_with_rates(manual_periods_df, algo_df)
    print(f"   匹配的漏水事件总数: {len(matched_events_df)}")
    
    # 5) 分析速率范围
    print("\n5. 分析漏水速率范围...")
    analyze_drain_rate_ranges(matched_events_df)

if __name__ == "__main__":
    main()