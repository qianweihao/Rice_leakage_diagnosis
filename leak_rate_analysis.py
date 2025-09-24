# -*- coding: utf-8 -*-
#漏水速率分析
import pandas as pd
import numpy as np

# 文件路径
DRAIN_EVENTS_FILE = "labeled_events_final_12h.csv"

# 目标设备
DEVICES = ["157B025010050", "157B025030023", "157B025030032"]

def load_drain_events():
    """加载漏水事件数据"""
    drain_df = pd.read_csv(DRAIN_EVENTS_FILE)
    drain_df['start'] = pd.to_datetime(drain_df['start'])
    drain_df['end'] = pd.to_datetime(drain_df['end'])
    return drain_df

def load_device_data(device_code):
    """加载设备数据"""
    file_path = f"label_data/device_{device_code}.csv"
    try:
        df = pd.read_csv(file_path)
        df['msgTime'] = pd.to_datetime(df['msgTime'])
        return df.sort_values('msgTime')
    except FileNotFoundError:
        print(f"警告：找不到设备 {device_code} 的数据文件")
        return pd.DataFrame()

def calculate_3hour_leak_rate(device_data, start_time):
    """计算从漏水事件开始3小时内的漏水速率"""
    end_time = start_time + pd.Timedelta(hours=3)
    
    # 获取时间窗口内的数据
    window_data = device_data[
        (device_data['msgTime'] >= start_time) & 
        (device_data['msgTime'] <= end_time)
    ]
    
    if len(window_data) < 2:
        return None, None
    
    # 计算液位变化
    start_level = window_data.iloc[0]['liquidLevelValue']
    end_level = window_data.iloc[-1]['liquidLevelValue']
    level_drop = start_level - end_level
    
    # 计算实际时间差（小时）
    actual_duration = (window_data.iloc[-1]['msgTime'] - window_data.iloc[0]['msgTime']).total_seconds() / 3600
    
    if actual_duration > 0 and level_drop > 0:
        leak_rate = level_drop / actual_duration
        return leak_rate, actual_duration
    
    return None, None

def analyze_leak_rates():
    """分析漏水速率"""
    drain_df = load_drain_events()
    
    print("=== 漏水速率详细分析 ===")
    print("计算方法1：基于12小时液位变化量 (delta_12h_mm / 12小时)")
    print("计算方法2：基于漏水事件开始后3小时实际液位变化")
    print("="*80)
    
    all_leak_rates = []
    all_leak_rates_3h = []
    
    for device_code in DEVICES:
        device_drains = drain_df[drain_df['code'] == device_code]
        
        if len(device_drains) == 0:
            print(f"\n设备 {device_code}: 无漏水事件")
            continue
            
        leak_rates = []
        leak_rates_3h = []
        device_data = load_device_data(device_code)
        
        print(f"\n设备 {device_code} 漏水事件详情:")
        print("-" * 120)
        print(f"{'序号':<4} {'开始时间':<16} {'结束时间':<16} {'持续时间':<8} {'液位降':<8} {'12h速率':<10} {'3h速率':<10} {'3h时长':<8}")
        print("-" * 120)
        
        for idx, (_, row) in enumerate(device_drains.iterrows(), 1):
            leak_rate_12h = None
            leak_rate_3h = None
            duration_3h = None
            
            # 计算12小时速率
            if 'delta_12h_mm' in row and pd.notna(row['delta_12h_mm']):
                leak_rate_12h = abs(row['delta_12h_mm']) / 12  # mm/h
                leak_rates.append(leak_rate_12h)
                all_leak_rates.append(leak_rate_12h)
            
            # 计算3小时速率
            if not device_data.empty:
                leak_rate_3h, duration_3h = calculate_3hour_leak_rate(device_data, row['start'])
                if leak_rate_3h is not None:
                    leak_rates_3h.append(leak_rate_3h)
                    all_leak_rates_3h.append(leak_rate_3h)
            
            duration_hours = (row['end'] - row['start']).total_seconds() / 3600
            
            # 格式化输出
            rate_12h_str = f"{leak_rate_12h:.2f}" if leak_rate_12h else "N/A"
            rate_3h_str = f"{leak_rate_3h:.2f}" if leak_rate_3h else "N/A"
            duration_3h_str = f"{duration_3h:.1f}h" if duration_3h else "N/A"
            
            print(f"{idx:<4} {row['start'].strftime('%m-%d %H:%M'):<16} "
                  f"{row['end'].strftime('%m-%d %H:%M'):<16} "
                  f"{duration_hours:.1f}h{'':<4} "
                  f"{abs(row['delta_12h_mm']) if pd.notna(row.get('delta_12h_mm')) else 'N/A':<8} "
                  f"{rate_12h_str:<10} "
                  f"{rate_3h_str:<10} "
                  f"{duration_3h_str:<8}")
        
        if leak_rates or leak_rates_3h:
            print("-" * 120)
            print(f"统计信息:")
            print(f"  事件总数: {len(leak_rates)}")
            
            if leak_rates:
                print(f"  12小时速率统计:")
                print(f"    平均速率: {np.mean(leak_rates):.2f} mm/h")
                print(f"    最大速率: {np.max(leak_rates):.2f} mm/h")
                print(f"    最小速率: {np.min(leak_rates):.2f} mm/h")
                print(f"    标准差:   {np.std(leak_rates):.2f} mm/h")
            
            if leak_rates_3h:
                print(f"  3小时速率统计:")
                print(f"    平均速率: {np.mean(leak_rates_3h):.2f} mm/h")
                print(f"    最大速率: {np.max(leak_rates_3h):.2f} mm/h")
                print(f"    最小速率: {np.min(leak_rates_3h):.2f} mm/h")
                print(f"    标准差:   {np.std(leak_rates_3h):.2f} mm/h")
                print(f"    有效事件: {len(leak_rates_3h)}/{len(device_drains)}")
    
    # 全局统计
    if all_leak_rates or all_leak_rates_3h:
        print("\n" + "="*80)
        print("全局漏水速率统计:")
        print("="*80)
        
        if all_leak_rates:
            print(f"12小时速率全局统计:")
            print(f"  总漏水事件数: {len(all_leak_rates)}")
            print(f"  全局平均速率: {np.mean(all_leak_rates):.2f} mm/h")
            print(f"  全局最大速率: {np.max(all_leak_rates):.2f} mm/h")
            print(f"  全局最小速率: {np.min(all_leak_rates):.2f} mm/h")
            print(f"  全局标准差:   {np.std(all_leak_rates):.2f} mm/h")
            
            # 12小时速率分布统计
            print("\n  12小时速率分布:")
            print(f"    < 0.3 mm/h:  {sum(1 for r in all_leak_rates if r < 0.3)} 事件 ({sum(1 for r in all_leak_rates if r < 0.3)/len(all_leak_rates)*100:.1f}%)")
            print(f"    0.3-0.6 mm/h: {sum(1 for r in all_leak_rates if 0.3 <= r < 0.6)} 事件 ({sum(1 for r in all_leak_rates if 0.3 <= r < 0.6)/len(all_leak_rates)*100:.1f}%)")
            print(f"    0.6-1.0 mm/h: {sum(1 for r in all_leak_rates if 0.6 <= r < 1.0)} 事件 ({sum(1 for r in all_leak_rates if 0.6 <= r < 1.0)/len(all_leak_rates)*100:.1f}%)")
            print(f"    >= 1.0 mm/h: {sum(1 for r in all_leak_rates if r >= 1.0)} 事件 ({sum(1 for r in all_leak_rates if r >= 1.0)/len(all_leak_rates)*100:.1f}%)")
        
        if all_leak_rates_3h:
            print(f"\n3小时速率全局统计:")
            print(f"  总有效事件数: {len(all_leak_rates_3h)}")
            print(f"  全局平均速率: {np.mean(all_leak_rates_3h):.2f} mm/h")
            print(f"  全局最大速率: {np.max(all_leak_rates_3h):.2f} mm/h")
            print(f"  全局最小速率: {np.min(all_leak_rates_3h):.2f} mm/h")
            print(f"  全局标准差:   {np.std(all_leak_rates_3h):.2f} mm/h")
            
            # 3小时速率分布统计
            print("\n  3小时速率分布:")
            print(f"    < 0.5 mm/h:  {sum(1 for r in all_leak_rates_3h if r < 0.5)} 事件 ({sum(1 for r in all_leak_rates_3h if r < 0.5)/len(all_leak_rates_3h)*100:.1f}%)")
            print(f"    0.5-1.0 mm/h: {sum(1 for r in all_leak_rates_3h if 0.5 <= r < 1.0)} 事件 ({sum(1 for r in all_leak_rates_3h if 0.5 <= r < 1.0)/len(all_leak_rates_3h)*100:.1f}%)")
            print(f"    1.0-2.0 mm/h: {sum(1 for r in all_leak_rates_3h if 1.0 <= r < 2.0)} 事件 ({sum(1 for r in all_leak_rates_3h if 1.0 <= r < 2.0)/len(all_leak_rates_3h)*100:.1f}%)")
            print(f"    >= 2.0 mm/h: {sum(1 for r in all_leak_rates_3h if r >= 2.0)} 事件 ({sum(1 for r in all_leak_rates_3h if r >= 2.0)/len(all_leak_rates_3h)*100:.1f}%)")
            
            print(f"\n速率对比分析:")
            if all_leak_rates:
                print(f"  12小时平均速率: {np.mean(all_leak_rates):.2f} mm/h")
            print(f"  3小时平均速率:  {np.mean(all_leak_rates_3h):.2f} mm/h")
            if all_leak_rates:
                ratio = np.mean(all_leak_rates_3h) / np.mean(all_leak_rates)
                print(f"  速率比值 (3h/12h): {ratio:.2f}")
                print(f"  说明: 3小时速率是12小时速率的 {ratio:.1f} 倍")

if __name__ == "__main__":
    analyze_leak_rates()