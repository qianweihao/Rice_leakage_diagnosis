#验证组-设备液位数据可视化脚本
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
DEVICES = ["157B025010050", "157B025030023", "157B025030032"]

# 设备ID到地块名称的映射
DEVICE_TO_PLOT_NAME = {
    "157B025010050": "H9",
    "157B025030023": "H10", 
    "157B025030032": "H14"
}

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

def plot_device_with_events(device_code, ax, anomalies_df, drain_df, weather_df, start_date=None, end_date=None, period_label=""):
    """绘制单个设备的数据和事件"""
    # 如果没有指定时间范围，使用默认值
    if start_date is None:
        start_date = pd.to_datetime('2025-08-10')
    if end_date is None:
        end_date = pd.to_datetime('2025-09-17')
    
    # 定义人工排水事件日期和级别
    manual_drain_events = {
        '157B025010050': {
            '2025-09-12': {'level': '中度排水', 'color': 'orange', 'linestyle': '--', 'marker': 'v'},
            '2025-09-16': {'level': '轻度排水', 'color': 'orange', 'linestyle': '-.', 'marker': 'o'}
        },
        '157B025030023': {
            '2025-08-13': {'level': '重度排水', 'color': 'orange', 'linestyle': ':', 'marker': 's'},
            '2025-09-12': {'level': '中度排水', 'color': 'orange', 'linestyle': '--', 'marker': 'v'},
            '2025-09-16': {'level': '轻度排水', 'color': 'orange', 'linestyle': '-.', 'marker': 'o'}
        },
        '157B025030032': {
            '2025-08-13': {'level': '重度排水', 'color': 'orange', 'linestyle': ':', 'marker': 's'},
            '2025-09-12': {'level': '中度排水', 'color': 'orange', 'linestyle': '--', 'marker': 'v'},
            '2025-09-16': {'level': '轻度排水', 'color': 'orange', 'linestyle': '-.', 'marker': 'o'}
        }
    }
    
    # 加载设备数据
    df = load_device_data(device_code)
    
    # 过滤时间范围
    df = df[(df['msgTime'] >= start_date) & (df['msgTime'] <= end_date)]
    
    # 创建双y轴
    ax2 = ax.twinx()
    
    # 绘制液位数据
    line1 = ax.plot(df['msgTime'], df['liquidLevel_clean'], 'b-', linewidth=1, alpha=0.7, label='液位值')
    
    # 绘制降雨量数据
    if not weather_df.empty:
        # 筛选在指定时间范围内的降雨数据
        weather_filtered = weather_df[
            (weather_df['date'] >= start_date) & 
            (weather_df['date'] <= end_date)
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
    
    # 移除异常事件相关代码
    device_anomalies = pd.DataFrame()  # 空的DataFrame，不显示异常事件
    
    # 筛选该设备的漏水事件（在时间范围内）
    device_drains = drain_df[
        (drain_df['code'] == device_code) &
        (drain_df['start'] <= end_date) &
        (drain_df['end'] >= start_date)
    ]
    
    # 定义需要展示算法漏水事件的指定日期
    target_leak_dates = ['2025-09-11', '2025-09-15', '2025-08-13', '2025-09-12']
    
    # 进一步筛选：只显示指定日期的漏水事件
    filtered_drains = []
    for _, row in device_drains.iterrows():
        event_date = row['start'].strftime('%Y-%m-%d')
        if event_date in target_leak_dates:
            filtered_drains.append(row)
    
    # 转换为DataFrame
    if filtered_drains:
        device_drains = pd.DataFrame(filtered_drains)
    else:
        device_drains = pd.DataFrame()
    
    # 标注漏水事件并计算漏水速率
    drain_labeled = False
    print(f"  漏水事件时间段:")
    for _, row in device_drains.iterrows():
        # 计算漏水速率 (mm/h)
        duration_hours = (row['end'] - row['start']).total_seconds() / 3600
        if 'delta_12h_mm' in row and pd.notna(row['delta_12h_mm']):
            # 使用12小时液位变化量计算速率
            leak_rate = abs(row['delta_12h_mm']) / 12  # mm/h
        else:
            # 如果没有delta_12h_mm，使用持续时间估算
            leak_rate = 5.0 / duration_hours if duration_hours > 0 else 0  # 假设平均漏水5mm
        
        label = '算法识别漏水' if not drain_labeled else ""
        # 增强算法漏水事件的可视化效果，添加网格线样式
        ax.axvspan(row['start'], row['end'], alpha=0.4, facecolor='red', label=label, 
                  edgecolor='darkred', linewidth=2, hatch='///')
        
        # 在时段中间添加文本标注
        mid_time = row['start'] + (row['end'] - row['start']) / 2
        y_pos = ax.get_ylim()[1] * 0.85
        ax.text(mid_time, y_pos, '漏水', 
               ha='center', va='center', fontsize=9, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
        
        # 输出漏水事件信息
        print(f"    {row['start'].strftime('%Y-%m-%d %H:%M')} 至 {row['end'].strftime('%Y-%m-%d %H:%M')} - 漏水速率: {leak_rate:.2f} mm/h")
        
        drain_labeled = True
    
    # 设置x轴显示范围
    ax.set_xlim(start_date, end_date)
    
    # 设置图表标题和标签
    plot_name = DEVICE_TO_PLOT_NAME.get(device_code, device_code)
    title_suffix = period_label if period_label else f"({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')})"
    ax.set_title(f'地块 {plot_name} 液位数据与事件分析 {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('液位值（cm）', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴日期 - 按小时标记
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 每12小时一个主刻度
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))   # 每6小时一个次刻度
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 标记人工排水事件
    device_drain_events = manual_drain_events.get(device_code, {})
    drain_legend_added = set()  # 记录已添加的排水级别图例
    
    for drain_date, event_info in device_drain_events.items():
        try:
            drain_datetime = pd.to_datetime(drain_date)
            # 检查日期是否在指定时间范围内
            if (drain_datetime >= start_date and 
                drain_datetime <= end_date):
                # 标记人工排水时间范围：从前一天中午12点到当天中午12点
                day_start = drain_datetime.replace(hour=12, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
                day_end = drain_datetime.replace(hour=12, minute=0, second=0, microsecond=0)
                
                # 绘制水平区域标记24小时排水时段
                ax.axvspan(day_start, day_end, 
                          color=event_info['color'], 
                          alpha=0.3, 
                          label=f"人工排水-{event_info['level']}" if event_info['level'] not in drain_legend_added else "")
                
                # 记录已添加的图例
                drain_legend_added.add(event_info['level'])
                
                # 在中间位置添加标记点（当天00:00）
                day_middle = drain_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
                y_max = ax.get_ylim()[1]
                ax.plot(day_middle, y_max * 0.95, 
                       marker=event_info['marker'], 
                       color=event_info['color'], 
                       markersize=10, 
                       markeredgecolor='black', 
                       markeredgewidth=1)
        except Exception as e:
            print(f"处理人工排水日期 {drain_date} 时出错: {e}")
    
    # 添加图例
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], color='b', linewidth=1, label='液位值'))
    
    # 添加算法识别漏水事件到图例
    if len(device_drains) > 0:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.4, 
                                           edgecolor='darkred', linewidth=1, hatch='///', label='算法识别漏水'))
    
    # 添加人工排水事件到图例（统一标识）
    device_drain_events = manual_drain_events.get(device_code, {})
    if device_drain_events:
        # 添加一个统一的人工排水图例项
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.3, 
                                           edgecolor='black', linewidth=1, label='人工排水事件'))
    
    # 添加降雨量到图例
    if not weather_df.empty:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='skyblue', alpha=0.6, label='日降雨量(cm)'))
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    # 打印事件统计
    print(f"设备 {device_code}:")
    print(f"  算法检测漏水事件数量（指定日期）: {len(device_drains)}")
    device_drain_events = manual_drain_events.get(device_code, {})
    print(f"  人工排水事件数量: {len(device_drain_events)}")
    if device_drain_events:
        print(f"  人工排水日期: {list(device_drain_events.keys())}")
    if len(device_drains) > 0:
        print(f"  显示的算法漏水事件日期: {[row['start'].strftime('%Y-%m-%d') for _, row in device_drains.iterrows()]}")
    print()

def calculate_leak_rate_statistics(drain_df):
    """计算漏水速率统计信息"""
    print("\n=== 漏水速率统计分析 ===")
    
    for device_code in DEVICES:
        device_drains = drain_df[drain_df['code'] == device_code]
        if len(device_drains) == 0:
            continue
            
        leak_rates = []
        print(f"\n设备 {device_code} 漏水速率详情:")
        
        for _, row in device_drains.iterrows():
            if 'delta_12h_mm' in row and pd.notna(row['delta_12h_mm']):
                leak_rate = abs(row['delta_12h_mm']) / 12  # mm/h
                leak_rates.append(leak_rate)
                
                duration_hours = (row['end'] - row['start']).total_seconds() / 3600
                print(f"  {row['start'].strftime('%m-%d %H:%M')} - {row['end'].strftime('%m-%d %H:%M')} | "
                      f"速率: {leak_rate:.2f} mm/h | 持续: {duration_hours:.1f}h | "
                      f"液位降: {abs(row['delta_12h_mm']):.1f}mm")
        
        if leak_rates:
            print(f"  平均漏水速率: {np.mean(leak_rates):.2f} mm/h")
            print(f"  最大漏水速率: {np.max(leak_rates):.2f} mm/h")
            print(f"  最小漏水速率: {np.min(leak_rates):.2f} mm/h")
            print(f"  漏水事件总数: {len(leak_rates)}")

def main():
    """主函数"""
    # 检查文件是否存在
    if not os.path.exists(ANOMALIES_FILE):
        print(f"错误：找不到文件 {ANOMALIES_FILE}")
        return
    
    if not os.path.exists(DRAIN_EVENTS_FILE):
        print(f"错误：找不到文件 {DRAIN_EVENTS_FILE}")
        return
    
    # 加载事件数据和气象数据
    anomalies_df, drain_df = load_events_data()
    weather_df = load_weather_data()
    
    # 计算漏水速率统计
    calculate_leak_rate_statistics(drain_df)
    
    # 定义不同设备的时间段
    device_periods = {
        '157B025010050': [
            (pd.to_datetime('2025-09-10'), pd.to_datetime('2025-09-17'), '(2025-09-10 至 2025-09-17)')
        ],
        '157B025030023': [
            (pd.to_datetime('2025-08-10'), pd.to_datetime('2025-08-14'), '(2025-08-10 至 2025-08-14)'),
            (pd.to_datetime('2025-09-10'), pd.to_datetime('2025-09-17'), '(2025-09-10 至 2025-09-17)')
        ],
        '157B025030032': [
            (pd.to_datetime('2025-08-10'), pd.to_datetime('2025-08-14'), '(2025-08-10 至 2025-08-14)'),
            (pd.to_datetime('2025-09-10'), pd.to_datetime('2025-09-17'), '(2025-09-10 至 2025-09-17)')
        ]
    }
    
    # 计算总的子图数量
    total_plots = sum(len(periods) for periods in device_periods.values())
    
    # 创建图形
    fig, axes = plt.subplots(total_plots, 1, figsize=(15, 6 * total_plots))
    fig.suptitle('地块液位数据及人工排水/算法识别排水事件标注', fontsize=16, fontweight='bold')
    
    # 如果只有一个子图，确保axes是列表
    if total_plots == 1:
        axes = [axes]
    
    # 绘制各设备不同时间段的数据
    plot_index = 0
    for device_code in DEVICES:
        periods = device_periods[device_code]
        for start_date, end_date, period_label in periods:
            try:
                plot_device_with_events(device_code, axes[plot_index], anomalies_df, drain_df, weather_df, 
                                       start_date, end_date, period_label)
            except FileNotFoundError:
                print(f"错误：找不到设备 {device_code} 的数据文件")
                axes[plot_index].text(0.5, 0.5, f'设备 {device_code} 数据文件不存在', 
                            transform=axes[plot_index].transAxes, ha='center', va='center', fontsize=14)
                axes[plot_index].set_title(f'设备 {device_code} (数据缺失) {period_label}', fontsize=14)
            plot_index += 1
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    
    # 保存图形
    plt.savefig('device_liquid_level_with_rainfall_val.png', dpi=300, bbox_inches='tight')
    print("\n图形已保存为 device_liquid_level_with_rainfall_val.png")

if __name__ == "__main__":
    main()