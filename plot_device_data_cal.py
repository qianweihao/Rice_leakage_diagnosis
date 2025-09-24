# -*- coding: utf-8 -*-
# #校准组-设备液位数据可视化脚本
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
LABEL_DATA_DIR = "label_data"
ANOMALIES_FILE = "labeled_anomalies.csv"
DRAIN_EVENTS_FILE = "labeled_events_final_12h.csv"
WEATHER_FILE = "峨桥气象数据0615_0915.xlsx"

# 目标设备
DEVICES = ["157B025030026", "157B025030033"]

def load_device_data(device_code):
    """加载设备数据"""
    file_path = f"{LABEL_DATA_DIR}/device_{device_code}.csv"
    df = pd.read_csv(file_path)
    df['msgTime'] = pd.to_datetime(df['msgTime'])
    return df

def load_events_data():
    """加载异常和漏水事件数据"""
    # 加载异常数据
    anomalies_df = pd.read_csv(ANOMALIES_FILE)
    anomalies_df['start'] = pd.to_datetime(anomalies_df['start'])
    anomalies_df['end'] = pd.to_datetime(anomalies_df['end'])
    
    # 加载漏水事件数据
    drain_df = pd.read_csv(DRAIN_EVENTS_FILE)
    drain_df['start'] = pd.to_datetime(drain_df['start'])
    drain_df['end'] = pd.to_datetime(drain_df['end'])
    
    return anomalies_df, drain_df

def load_weather_data():
    """加载气象数据，获取每天的累积降雨量"""
    try:
        df = pd.read_excel(WEATHER_FILE)
        df['time'] = pd.to_datetime(df['time'])
        df['date'] = df['time'].dt.date
        # 获取每天最后一条记录的sum_rain作为当天累积降雨量
        daily_rain = df.groupby('date')['sum_rain'].last().reset_index()
        daily_rain['date'] = pd.to_datetime(daily_rain['date'])
        return daily_rain
    except FileNotFoundError:
        print(f"气象数据文件 {WEATHER_FILE} 不存在")
        return pd.DataFrame()
    except Exception as e:
        print(f"读取气象数据时出错: {e}")
        return pd.DataFrame()

def plot_device_with_events(device_code, ax, anomalies_df, drain_df, weather_df):
    """绘制单个设备的数据和事件"""
    # 定义人工排水事件日期
    manual_drain_dates = [
        '2025-07-08',
        '2025-07-09', 
        '2025-07-26',
        '2025-07-27'
    ]
    
    # 加载设备数据
    df = load_device_data(device_code)
    
    # 创建双y轴
    ax2 = ax.twinx()
    
    # 绘制液位数据
    line1 = ax.plot(df['msgTime'], df['liquidLevelValue'], 'b-', linewidth=1, alpha=0.7, label='液位值')
    
    # 绘制降雨量数据
    if not weather_df.empty:
        # 筛选在液位数据时间范围内的降雨数据
        start_date = df['msgTime'].min().date()
        end_date = df['msgTime'].max().date()
        weather_filtered = weather_df[
            (weather_df['date'].dt.date >= start_date) & 
            (weather_df['date'].dt.date <= end_date)
        ]
        
        if not weather_filtered.empty:
            # 先将降雨量除以10，再筛选有降雨的日期
            rain_data = weather_filtered.copy()
            rain_data['sum_rain'] = rain_data['sum_rain'] / 10
            rain_data = rain_data[rain_data['sum_rain'] > 0]
            if not rain_data.empty:
                bars = ax2.bar(rain_data['date'], rain_data['sum_rain'], 
                              alpha=0.6, color='skyblue', width=0.8, label='日降雨量(cm)')
                ax2.set_ylabel('降雨量 (cm)', fontsize=12, color='skyblue')
                ax2.tick_params(axis='y', labelcolor='skyblue')
                
                # 设置降雨量Y轴范围与液位值Y轴保持一致
                liquid_ylim = ax.get_ylim()
                ax2.set_ylim(liquid_ylim)
    
    # 筛选该设备的异常事件
    device_anomalies = anomalies_df[anomalies_df['code'] == device_code]
    
    # 标注异常事件
    anomaly_labeled = False
    for _, row in device_anomalies.iterrows():
        label = '异常' if not anomaly_labeled else ""
        ax.axvspan(row['start'], row['end'], alpha=0.3, color='red', label=label)
        anomaly_labeled = True
    
    # 筛选该设备的漏水事件
    device_drains = drain_df[drain_df['code'] == device_code]
    
    # 标注漏水事件
    drain_labeled = False
    for _, row in device_drains.iterrows():
        label = '漏水' if not drain_labeled else ""
        ax.axvspan(row['start'], row['end'], alpha=0.3, color='orange', label=label)
        drain_labeled = True
    
    # 设置标题和标签
    ax.set_title(f'设备 {device_code} 液位数据', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('液位值（cm）', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 标记人工排水事件
    for drain_date in manual_drain_dates:
        try:
            drain_datetime = pd.to_datetime(drain_date)
            # 检查日期是否在数据范围内
            if (drain_datetime >= df['msgTime'].min().normalize() and 
                drain_datetime <= df['msgTime'].max().normalize()):
                ax.axvline(x=drain_datetime, color='green', linestyle='--', linewidth=2, alpha=0.7)
        except Exception as e:
            print(f"处理人工排水日期 {drain_date} 时出错: {e}")
    
    # 添加图例
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], color='b', linewidth=1, label='液位值'))
    
    if len(device_anomalies) > 0:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.3, label='异常'))
    
    if len(device_drains) > 0:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.3, label='漏水'))
    
    # 添加人工排水事件到图例
    legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='人工排水事件'))
    
    # 添加降雨量到图例
    if not weather_df.empty:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', alpha=0.6, label='日降雨量(cm)'))
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    # 打印事件统计
    print(f"设备 {device_code}:")
    print(f"  异常事件数量: {len(device_anomalies)}")
    print(f"  漏水事件数量: {len(device_drains)}")
    if len(device_drains) > 0:
        print(f"  漏水事件时间段:")
        for _, row in device_drains.iterrows():
            print(f"    {row['start'].strftime('%Y-%m-%d %H:%M')} 至 {row['end'].strftime('%Y-%m-%d %H:%M')}")
    print()

def main():
    """主函数"""
    # 检查文件是否存在
    if not os.path.exists(ANOMALIES_FILE):
        print(f"错误：找不到文件 {ANOMALIES_FILE}")
        return
    
    if not os.path.exists(DRAIN_EVENTS_FILE):
        print(f"错误：找不到文件 {DRAIN_EVENTS_FILE}")
        return
    
    # 定义人工排水事件日期
    manual_drain_dates = [
        '2025-07-07',
        '2025-07-08', 
        '2025-07-26'
    ]
    
    # 加载事件数据和气象数据
    anomalies_df, drain_df = load_events_data()
    weather_df = load_weather_data()
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('设备液位数据及异常/漏水事件标注', fontsize=16, fontweight='bold')
    
    # 绘制两个设备的数据
    for i, device_code in enumerate(DEVICES):
        try:
            plot_device_with_events(device_code, axes[i], anomalies_df, drain_df, weather_df)
        except FileNotFoundError:
            print(f"错误：找不到设备 {device_code} 的数据文件")
            axes[i].text(0.5, 0.5, f'设备 {device_code} 数据文件不存在', 
                        transform=axes[i].transAxes, ha='center', va='center', fontsize=14)
            axes[i].set_title(f'设备 {device_code} (数据缺失)', fontsize=14)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存图形
    plt.savefig('device_liquid_level_with_rainfall_cal.png', dpi=300, bbox_inches='tight')
    print("图形已保存为 device_liquid_level_with_rainfall_cal.png")

if __name__ == "__main__":
    main()