#过程图像分析
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
device_id = '157B025030039'
print(f"分析设备 {device_id} 的漏水事件")

# 配置API参数
URL = "https://iland.zoomlion.com/open-sharing-platform/zlapi/irrigationApi/v1/getZnjsWaterHisByDeviceCode"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "dWCkcdbdSeMqHyMQmZruWzwHR30cspVH"
}

# 获取原始数据函数
def fetch_raw_data(device_code, start_day="2025-06-15", end_day="2025-09-15"):
    payload = {
        "deviceCode": device_code,
        "startDay": start_day,
        "endDay": end_day
    }
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        resp = session.post(URL, headers=HEADERS, json=payload, timeout=120)
        if resp.status_code == 200:
            js = resp.json()
            data = js.get("data", [])
            return pd.DataFrame(data)
        else:
            print(f"API请求失败，状态码: {resp.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"获取原始数据失败: {e}")
        return pd.DataFrame()

# 读取清洗后的数据
device_data = pd.read_csv(f'clean_results_smooth/device_{device_id}.csv')
device_data['msgTime'] = pd.to_datetime(device_data['msgTime'])

# 获取原始数据
raw_data = fetch_raw_data(device_id)
if not raw_data.empty:
    # 处理原始数据的时间列
    time_col = None
    for col in ["msgTimeStr", "msgTime", "time"]:
        if col in raw_data.columns:
            time_col = col
            break
    
    if time_col:
        def parse_time_col(series):
            def _p(v):
                try:
                    if isinstance(v, (int, float)):
                        s = str(int(v))
                    else:
                        s = str(v).strip()
                    if s.isdigit():
                        if len(s) == 13:
                            return pd.to_datetime(int(s), unit="ms")
                        elif len(s) == 10:
                            return pd.to_datetime(int(s), unit="s")
                    return pd.to_datetime(s, errors="coerce")
                except Exception:
                    return pd.NaT
            return series.map(_p)
        
        raw_data["msgTime"] = parse_time_col(raw_data[time_col])
        raw_data = raw_data.dropna(subset=["msgTime"])
        raw_data.sort_values("msgTime", inplace=True)
        
        # 处理液位数据
        if "liquidLevelValue" in raw_data.columns:
            raw_data["liquidLevelValue"] = pd.to_numeric(raw_data["liquidLevelValue"], errors="coerce")
            # 标记异常值（>24mm）
            raw_data["is_outlier"] = raw_data["liquidLevelValue"] > 24
        
        print(f"成功获取原始数据: {len(raw_data)}条记录")
    else:
        print("原始数据中未找到时间列")
        raw_data = pd.DataFrame()
else:
    print("未能获取到原始数据，将使用清洗后数据")
    raw_data = pd.DataFrame()

# 读取标注事件
labeled_events = pd.read_csv('labeled_events_final_12h.csv')
device_events = labeled_events[labeled_events['code'] == device_id]

print(f"找到 {len(device_events)} 个标注的漏水事件")
for _, event in device_events.iterrows():
    print(f"事件: {event['start']} - {event['end']}, 液位变化: {event['delta_12h_mm']:.2f}mm")

# 读取未经12h后置确认的事件（用于对比）
labeled_events_raw = pd.read_csv('labeled_events_updated.csv')
device_events_raw = labeled_events_raw[(labeled_events_raw['code'] == device_id) & (labeled_events_raw['label'] == 'drain')]

# 创建图形
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 24))
fig.suptitle(f'设备 {device_id} 漏水事件分析', fontsize=16, fontweight='bold', y=0.98)

# 第一个图：展示快速下降 vs 短窗累计下降
ax1.plot(device_data['msgTime'], device_data['liquidLevel_clean'], 'b-', linewidth=1, alpha=0.7, label='液位数据')

# 标记漏水事件
for _, event in device_events.iterrows():
    start_time = pd.to_datetime(event['start'])
    end_time = pd.to_datetime(event['end'])
    
    # 获取事件期间的数据
    event_mask = (device_data['msgTime'] >= start_time) & (device_data['msgTime'] <= end_time)
    event_data = device_data[event_mask]
    
    if len(event_data) > 0:
        # 计算事件特征
        duration_hours = (end_time - start_time).total_seconds() / 3600
        total_drop = event['delta_12h_mm']
        avg_rate = total_drop / duration_hours if duration_hours > 0 else 0
        
        # 分类：多小时快速下降 vs 短窗累计下降
        if duration_hours >= 8 and avg_rate < -0.5:  # 多小时快速下降
            color = 'red'
            label_text = f'多小时快速下降\n{duration_hours:.1f}h, {avg_rate:.2f}mm/h'
            event_type = '多小时快速下降'
        else:  # 短窗累计下降
            color = 'orange'
            label_text = f'短窗累计下降\n{duration_hours:.1f}h, {total_drop:.1f}mm'
            event_type = '短窗累计下降'
        
        # 绘制事件区域
        ax1.axvspan(start_time, end_time, alpha=0.3, color=color)
        
        # 添加标注
        mid_time = start_time + (end_time - start_time) / 2
        mid_level = event_data['liquidLevel_clean'].mean()
        ax1.annotate(label_text, xy=(mid_time, mid_level), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    fontsize=8, ha='left')
        
        print(f"事件分类: {event_type}, 时长: {duration_hours:.1f}小时, 平均速率: {avg_rate:.2f}mm/h")

ax1.set_xlabel('时间')
ax1.set_ylabel('液位 (mm)')
ax1.set_title('图1: 漏水事件分类 - 多小时快速下降 vs 短窗累计下降')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 设置时间轴格式
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 第二个图：展示三重过滤机制
ax2.plot(device_data['msgTime'], device_data['liquidLevel_clean'], 'b-', linewidth=1, alpha=0.7, label='液位数据')

# 标记mask_final为True的点（通过过滤的漏水点）
mask_true = device_data['mask_final'] == True
if mask_true.any():
    ax2.scatter(device_data.loc[mask_true, 'msgTime'], 
               device_data.loc[mask_true, 'liquidLevel_clean'], 
               c='red', s=20, alpha=0.8, label='通过过滤的漏水点', zorder=5)

# 模拟三重过滤的标记（基于实际算法逻辑）
for _, event in device_events.iterrows():
    start_time = pd.to_datetime(event['start'])
    end_time = pd.to_datetime(event['end'])
    
    # 获取事件前后的数据用于过滤分析
    extended_start = start_time - timedelta(hours=6)
    extended_end = end_time + timedelta(hours=6)
    
    extended_mask = (device_data['msgTime'] >= extended_start) & (device_data['msgTime'] <= extended_end)
    extended_data = device_data[extended_mask].copy()
    
    if len(extended_data) > 5:
        # 模拟过滤逻辑
        event_start_idx = extended_data[extended_data['msgTime'] >= start_time].index[0] if len(extended_data[extended_data['msgTime'] >= start_time]) > 0 else None
        
        if event_start_idx is not None:
            event_row = extended_data.loc[event_start_idx]
            
            # 过滤器1：突变过滤
            pre_5h_data = extended_data[extended_data['msgTime'] <= start_time - timedelta(hours=5)]
            if len(pre_5h_data) > 0:
                pre_5h_level = pre_5h_data['liquidLevel_clean'].iloc[-1]
                sudden_change = abs(event_row['liquidLevel_clean'] - pre_5h_level)
                filter1_triggered = sudden_change > 5.0
            else:
                filter1_triggered = False
            
            # 过滤器2：高点过滤
            pre_1h_data = extended_data[extended_data['msgTime'] <= start_time - timedelta(hours=1)]
            if len(pre_1h_data) > 0 and len(pre_5h_data) > 0:
                pre_1h_level = pre_1h_data['liquidLevel_clean'].iloc[-1]
                filter2_triggered = (pre_1h_level > event_row['liquidLevel_clean'] and 
                                   pre_1h_level > pre_5h_level)
            else:
                filter2_triggered = False
            
            # 过滤器3：时间关联过滤（简化版）
            filter3_triggered = False  # 需要异常事件数据来判断
            
            # 标记过滤结果
            filter_status = []
            if filter1_triggered:
                filter_status.append('突变过滤')
            if filter2_triggered:
                filter_status.append('高点过滤')
            if filter3_triggered:
                filter_status.append('时间关联过滤')
            
            # 绘制过滤标记
            if filter_status:
                filter_text = '\n'.join(filter_status)
                ax2.axvspan(start_time, end_time, alpha=0.4, color='purple')
                
                mid_time = start_time + (end_time - start_time) / 2
                ax2.annotate(f'被过滤\n{filter_text}', 
                           xy=(mid_time, event_row['liquidLevel_clean']), 
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.7),
                           fontsize=8, ha='left', color='white')
                
                print(f"事件 {start_time} 被过滤: {filter_status}")
            else:
                # 未被过滤的事件
                ax2.axvspan(start_time, end_time, alpha=0.3, color='green')
                
                mid_time = start_time + (end_time - start_time) / 2
                ax2.annotate('通过过滤', 
                           xy=(mid_time, event_row['liquidLevel_clean']), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           fontsize=8, ha='left', color='white')
                
                print(f"事件 {start_time} 通过所有过滤器")

ax2.set_xlabel('时间')
ax2.set_ylabel('液位 (mm)')
ax2.set_title('图2: 三重过滤机制标记')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 设置时间轴格式
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 第三个图：展示12小时后置确认机制
ax3.plot(device_data['msgTime'], device_data['liquidLevel_clean'], 'b-', linewidth=1, alpha=0.7, label='液位数据')

# 标记未经12h确认的drain事件（灰色）
for _, event in device_events_raw.iterrows():
    start_time = pd.to_datetime(event['start'])
    end_time = pd.to_datetime(event['end'])
    ax3.axvspan(start_time, end_time, alpha=0.2, color='gray', label='未确认事件' if _ == device_events_raw.index[0] else "")
    
    # 添加标注
    mid_time = start_time + (end_time - start_time) / 2
    event_mask = (device_data['msgTime'] >= start_time) & (device_data['msgTime'] <= end_time)
    event_data = device_data[event_mask]
    if len(event_data) > 0:
        mid_level = event_data['liquidLevel_clean'].mean()
        ax3.annotate('未确认', xy=(mid_time, mid_level), 
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='gray', alpha=0.5),
                    fontsize=7, ha='left')

# 标记通过12h确认的drain事件（绿色）
for _, event in device_events.iterrows():
    start_time = pd.to_datetime(event['start'])
    end_time = pd.to_datetime(event['end'])
    
    # 绘制确认事件区域
    ax3.axvspan(start_time, end_time, alpha=0.4, color='green', label='12h确认事件' if _ == device_events.index[0] else "")
    
    # 标记12h后的时间点和液位变化
    t12 = start_time + pd.Timedelta(hours=12)
    delta_12h = event['delta_12h_mm']
    
    # 添加12h确认标记
    ax3.axvline(x=t12, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 获取事件期间的数据
    event_mask = (device_data['msgTime'] >= start_time) & (device_data['msgTime'] <= end_time)
    event_data = device_data[event_mask]
    
    if len(event_data) > 0:
        mid_time = start_time + (end_time - start_time) / 2
        mid_level = event_data['liquidLevel_clean'].mean()
        
        # 添加确认标注
        ax3.annotate(f'12h确认\n下降{abs(delta_12h):.1f}mm', 
                    xy=(mid_time, mid_level), 
                    xytext=(10, 15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                    fontsize=8, ha='left', color='white')
        
        # 标注12h时间点
        t12_data = device_data[device_data['msgTime'] >= t12]
        if len(t12_data) > 0:
            t12_level = t12_data['liquidLevel_clean'].iloc[0]
            ax3.annotate(f'+12h\n{t12_level:.1f}mm', 
                        xy=(t12, t12_level), 
                        xytext=(5, -15), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7),
                        fontsize=7, ha='center', color='white')

ax3.set_xlabel('时间')
ax3.set_ylabel('液位 (mm)')
ax3.set_title('图3: 12小时后置确认机制 - 确认前后事件对比')
ax3.grid(True, alpha=0.3)
ax3.legend()

# 设置时间轴格式
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 第四个图：原始水位数据及异常值标记
if not raw_data.empty and 'liquidLevelValue' in raw_data.columns:
    # 绘制原始水位数据
    ax4.plot(raw_data['msgTime'], raw_data['liquidLevelValue'], 'blue', linewidth=0.8, alpha=0.7, label='原始水位数据')
    
    # 标记异常值点（基于is_outlier列）
    if 'is_outlier' in raw_data.columns:
        outliers = raw_data[raw_data['is_outlier'] == True]
        if not outliers.empty:
            ax4.scatter(outliers['msgTime'], outliers['liquidLevelValue'], 
                       color='red', s=40, alpha=0.8, label=f'异常水位点 ({len(outliers)}个)', zorder=5)
            
            # 为异常点添加标注（仅标注前几个，避免过于拥挤）
            outlier_sample = outliers.head(5)  # 只标注前5个异常点
            for idx, row in outlier_sample.iterrows():
                ax4.annotate(f'{row["liquidLevelValue"]:.1f}mm', 
                           xy=(row['msgTime'], row['liquidLevelValue']), 
                           xytext=(5, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7),
                           fontsize=7, ha='left', color='white')
    
    # 叠加显示清洗后的数据进行对比
    if 'liquidLevel_clean' in device_data.columns:
        ax4.plot(device_data['msgTime'], device_data['liquidLevel_clean'], 
                'green', linewidth=1.2, alpha=0.8, label='清洗后数据', zorder=3)
else:
    # 如果没有原始数据，使用清洗后数据中的原始列
    if 'liquidLevel' in device_data.columns:
        ax4.plot(device_data['msgTime'], device_data['liquidLevel'], 'blue', linewidth=1, label='原始水位数据')
        
        if 'is_outlier' in device_data.columns:
            outliers = device_data[device_data['is_outlier'] == 1]
            if not outliers.empty:
                ax4.scatter(outliers['msgTime'], outliers['liquidLevel'], 
                           color='red', s=40, alpha=0.8, label=f'异常水位点 ({len(outliers)}个)', zorder=5)
    else:
        ax4.text(0.5, 0.5, '无原始水位数据', transform=ax4.transAxes, 
                 ha='center', va='center', fontsize=14, color='gray')

ax4.set_xlabel('时间')
ax4.set_ylabel('液位 (mm)')
ax4.set_title('图4: 原始水位数据及异常值标记')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 设置时间轴格式
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

# 调整布局，增加子图间距
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)

# 保存图形
output_file = f'device_{device_id}_leak_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n图形已保存为: {output_file}")

# 显示图形
plt.show()

# 输出统计信息
print("\n=== 分析总结 ===")
print(f"设备: {device_id}")
print(f"未经12h确认的drain事件数: {len(device_events_raw)}")
print(f"通过12h确认的drain事件数: {len(device_events)}")
print(f"12h确认通过率: {len(device_events)/len(device_events_raw)*100:.1f}%" if len(device_events_raw) > 0 else "12h确认通过率: 0%")

# 统计事件类型
long_duration_events = 0
short_duration_events = 0

for _, event in device_events.iterrows():
    start_time = pd.to_datetime(event['start'])
    end_time = pd.to_datetime(event['end'])
    duration_hours = (end_time - start_time).total_seconds() / 3600
    total_drop = event['delta_12h_mm']
    avg_rate = total_drop / duration_hours if duration_hours > 0 else 0
    
    if duration_hours >= 8 and avg_rate < -0.5:
        long_duration_events += 1
    else:
        short_duration_events += 1

print(f"多小时快速下降事件: {long_duration_events}")
print(f"短窗累计下降事件: {short_duration_events}")
print(f"通过过滤的数据点数: {mask_true.sum()}")

print(f"\n=== 数据质量统计 ===")
if not raw_data.empty:
    print(f"原始数据点: {len(raw_data)}个")
    if 'is_outlier' in raw_data.columns:
        outlier_count = len(raw_data[raw_data['is_outlier'] == True])
        print(f"异常值数量: {outlier_count}个")
        print(f"异常值比例: {outlier_count/len(raw_data)*100:.2f}%")
    else:
        print(f"异常值数量: 原始数据中无异常值标记")
else:
    print(f"未获取到原始数据")

print(f"清洗后数据点: {len(device_data)}个")
if 'is_outlier' in device_data.columns:
    outlier_count_clean = len(device_data[device_data['is_outlier'] == 1])
    print(f"清洗后异常值数量: {outlier_count_clean}个")
    print(f"清洗后异常值比例: {outlier_count_clean/len(device_data)*100:.2f}%")
else:
    print(f"清洗后异常值数量: 数据中无异常值标记")

# 12h确认详细信息
if len(device_events) > 0:
    print("\n=== 12小时确认详情 ===")
    for _, event in device_events.iterrows():
        print(f"事件: {event['start']} - {event['end']}")
        print(f"  起始液位: {event['level_start_mm']:.1f}mm")
        print(f"  12h后液位: {event['level_at_12h_mm']:.1f}mm")
        print(f"  12h液位变化: {event['delta_12h_mm']:.2f}mm")
        print(f"  确认状态: {'通过' if event['delta_12h_mm'] <= -2.5 else '未通过'}")