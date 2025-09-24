# -*- coding: utf-8 -*-
# 计算算法排水检测的全面评估指标
import os
import glob
import pandas as pd
from datetime import timedelta

# ===================== 配置 =====================
LABEL_DATA_DIR     = "label_data"
ALGO_RESULTS_FILE  = "labeled_events_final_12h.csv"  
PATTERN            = "device_*.csv"
IOU_THRESHOLD      = 0.5   # IoU匹配阈值
MERGE_GAP_HOURS    = 2     # 人工标记合并间隔（小时）

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
    加载算法检测结果（CSV 需含列：code,start,end,label）
    返回: DataFrame[code, start, end] —— 仅保留 label=='drain'
    """
    if not os.path.exists(ALGO_RESULTS_FILE):
        print(f"算法结果文件 {ALGO_RESULTS_FILE} 不存在")
        return pd.DataFrame()

    df = pd.read_csv(ALGO_RESULTS_FILE)
    if not {'code','start','end','label'}.issubset(df.columns):
        print(f"算法结果文件缺少必要列（需要 code,start,end,label）")
        return pd.DataFrame()

    df = df[df['label'] == 'drain'].copy()
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end']   = pd.to_datetime(df['end'],   errors='coerce')
    df = df.dropna(subset=['start','end','code'])

    return df[['code','start','end']]

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

# ===================== 时间重叠计算 =====================
def calculate_iou(start1, end1, start2, end2):
    """
    计算两个时间段的IoU（Intersection over Union）
    """
    # 交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection = (intersection_end - intersection_start).total_seconds()
    
    # 并集
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = (union_end - union_start).total_seconds()
    
    return intersection / union if union > 0 else 0.0

def has_overlap(start1, end1, start2, end2):
    """
    判断两个时间段是否有重叠
    """
    return max(start1, start2) < min(end1, end2)

def is_strictly_contained(inner_start, inner_end, outer_start, outer_end):
    """
    判断内部时间段是否被外部时间段严格包含
    """
    return outer_start <= inner_start and inner_end <= outer_end

# ===================== 全面评估指标计算 =====================
def calculate_comprehensive_metrics(manual_points_df, algo_df, match_strategy='overlap', iou_threshold=IOU_THRESHOLD):
    """
    计算全面的评估指标
    
    参数:
        match_strategy: 'strict' (严格包含), 'overlap' (时间重叠), 'iou' (IoU阈值)
        iou_threshold: IoU匹配阈值（仅在match_strategy='iou'时使用）
    
    返回:
        results（各设备统计列表）, overall_metrics
    """
    if manual_points_df.empty:
        print("没有人工标记的排水事件点（is_outlier=1）")
        return [], {}

    if algo_df.empty:
        print("没有算法检测的排水事件")
        return [], {}

    # 合并人工事件
    manual_periods_df = get_manual_drain_periods(manual_points_df)
    total_manual_periods = len(manual_periods_df)
    print(f"   人工标记排水【时间段】总数: {total_manual_periods}")

    # 设备集合
    all_devices = set(manual_periods_df['code'].unique()) | set(algo_df['code'].unique())
    
    # 全局统计
    global_tp = 0  # True Positive
    global_fp = 0  # False Positive  
    global_fn = 0  # False Negative
    
    results = []
    
    for device in sorted(all_devices):
        m_dev = manual_periods_df[manual_periods_df['code'] == device]
        a_dev = algo_df[algo_df['code'] == device]

        device_manual_periods = len(m_dev)
        device_algo = len(a_dev)
        
        # 计算TP, FP, FN
        tp = 0  # 算法正确检测的事件数
        fp = 0  # 算法误检的事件数
        fn = 0  # 算法漏检的事件数
        
        # 标记已匹配的人工事件和算法事件
        matched_manual = set()
        matched_algo = set()
        
        # 对每个算法事件，寻找匹配的人工事件
        for algo_idx, ar in a_dev.iterrows():
            a_start, a_end = ar['start'], ar['end']
            found_match = False
            
            for manual_idx, mr in m_dev.iterrows():
                m_start, m_end = mr['start'], mr['end']
                
                # 根据匹配策略判断是否匹配
                is_match = False
                if match_strategy == 'strict':
                    is_match = is_strictly_contained(a_start, a_end, m_start, m_end)
                elif match_strategy == 'overlap':
                    is_match = has_overlap(a_start, a_end, m_start, m_end)
                elif match_strategy == 'iou':
                    iou = calculate_iou(a_start, a_end, m_start, m_end)
                    is_match = iou >= iou_threshold
                
                if is_match and manual_idx not in matched_manual:
                    found_match = True
                    matched_manual.add(manual_idx)
                    matched_algo.add(algo_idx)
                    break
            
            if found_match:
                tp += 1
            else:
                fp += 1
        
        # 计算漏检：未被匹配的人工事件
        fn = device_manual_periods - len(matched_manual)
        
        # 计算各项指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        # 对于FPR，我们假设TN=0（因为很难定义真正的负样本时间段）
        # 所以FPR = FP / (FP + TN) ≈ FP / FP = 1 (当FP>0时)
        false_positive_rate = fp / device_algo if device_algo > 0 else 0
        
        results.append({
            'device': device,
            'manual_periods': device_manual_periods,
            'algo_count': device_algo,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'miss_rate': miss_rate,
            'false_positive_rate': false_positive_rate,
            'match_strategy': match_strategy
        })
        
        global_tp += tp
        global_fp += fp
        global_fn += fn
    
    # 计算总体指标
    overall_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    overall_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_miss_rate = global_fn / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    
    overall_metrics = {
        'total_manual_periods': total_manual_periods,
        'total_algo_events': len(algo_df),
        'global_tp': global_tp,
        'global_fp': global_fp,
        'global_fn': global_fn,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'overall_miss_rate': overall_miss_rate,
        'match_strategy': match_strategy
    }
    
    return results, overall_metrics

# ===================== 主流程 =====================
def main():
    print("=== 排水检测全面评估指标计算 ===")

    # 1) 加载人工标记
    print("\n1. 加载人工标记数据...")
    manual_df = load_manual_labels()
    print(f"   人工标记排水【时间点】总数: {len(manual_df)}")

    # 2) 加载算法结果
    print("\n2. 加载算法检测结果...")
    algo_df = load_algo_results()
    print(f"   算法检测排水事件总数: {len(algo_df)}")

    if manual_df.empty or algo_df.empty:
        print("\n数据不足，无法计算评估指标")
        return

    # 3) 使用不同匹配策略计算指标
    strategies = [
        ('strict', '严格包含'),
        ('overlap', '时间重叠'),
        ('iou', f'IoU≥{IOU_THRESHOLD}')
    ]
    
    all_results = []
    
    for strategy, strategy_name in strategies:
        print(f"\n3. 计算评估指标 - {strategy_name}...")
        
        if strategy == 'iou':
            results, overall_metrics = calculate_comprehensive_metrics(manual_df, algo_df, strategy, IOU_THRESHOLD)
        else:
            results, overall_metrics = calculate_comprehensive_metrics(manual_df, algo_df, strategy)
        
        if not results:
            continue
            
        results_df = pd.DataFrame(results)
        
        # 输出结果
        print(f"\n=== {strategy_name} - 各设备评估指标 ===")
        print(f"{'设备':<15} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'漏检率':<8} {'误检率':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 90)
        
        for _, row in results_df.sort_values('f1_score', ascending=False).iterrows():
            print(f"{row['device']:<15} {row['precision']:<8.3f} {row['recall']:<8.3f} {row['f1_score']:<8.3f} "
                  f"{row['miss_rate']:<8.3f} {row['false_positive_rate']:<8.3f} "
                  f"{row['tp']:<4} {row['fp']:<4} {row['fn']:<4}")
        
        print(f"\n=== {strategy_name} - 总体指标 ===")
        print(f"总体精确率: {overall_metrics['overall_precision']:.3f}")
        print(f"总体召回率: {overall_metrics['overall_recall']:.3f}")
        print(f"总体F1分数: {overall_metrics['overall_f1']:.3f}")
        print(f"总体漏检率: {overall_metrics['overall_miss_rate']:.3f}")
        print(f"总体误检率: {overall_metrics['global_fp']/(overall_metrics['global_tp']+overall_metrics['global_fp']):.3f}" if (overall_metrics['global_tp']+overall_metrics['global_fp']) > 0 else "总体误检率: 0.000")
        print(f"人工排水时间段总数: {overall_metrics['total_manual_periods']}")
        print(f"算法检测事件总数: {overall_metrics['total_algo_events']}")
        print(f"全局TP: {overall_metrics['global_tp']}, FP: {overall_metrics['global_fp']}, FN: {overall_metrics['global_fn']}")
        
        # 如果是时间重叠策略，额外突出显示
        if strategy == 'overlap':
            print("\n" + "="*60)
            print("【时间重叠策略 - 重点关注指标】")
            print("="*60)
            print(f"[+] 总体精确率: {overall_metrics['overall_precision']:.3f} (算法检测正确率)")
            print(f"[+] 总体召回率: {overall_metrics['overall_recall']:.3f} (人工标记覆盖率)")
            print(f"[+] 总体F1分数: {overall_metrics['overall_f1']:.3f} (综合评估指标)")
            print(f"[-] 总体漏检率: {overall_metrics['overall_miss_rate']:.3f} (未检测到的比例)")
            false_positive_rate = overall_metrics['global_fp']/(overall_metrics['global_tp']+overall_metrics['global_fp']) if (overall_metrics['global_tp']+overall_metrics['global_fp']) > 0 else 0
            print(f"[-] 总体误检率: {false_positive_rate:.3f} (误报比例)")
            print(f"[*] 检测效果: {overall_metrics['global_tp']}个正确检测 / {overall_metrics['total_manual_periods']}个实际排水事件")
            print(f"[*] 误报情况: {overall_metrics['global_fp']}个误报 / {overall_metrics['total_algo_events']}个算法检测")
            print("="*60)
        
        # 添加策略标识
        results_df['strategy'] = strategy_name
        all_results.append(results_df)
    
    # 4) 保存详细结果
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        output_file = "drain_accuracy_comprehensive.csv"
        combined_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {output_file}")
        
        # 5) 额外统计分析
        print("\n=== 统计分析 ===")
        for strategy, strategy_name in strategies:
            strategy_data = combined_results[combined_results['strategy'] == strategy_name]
            if not strategy_data.empty:
                devices_with_algo = strategy_data[strategy_data['algo_count'] > 0]
                if not devices_with_algo.empty:
                    avg_precision = devices_with_algo['precision'].mean()
                    avg_recall = devices_with_algo['recall'].mean()
                    avg_f1 = devices_with_algo['f1_score'].mean()
                    high_f1_devices = (devices_with_algo['f1_score'] >= 0.8).sum()
                    total_devices_with_algo = len(devices_with_algo)
                    
                    print(f"\n{strategy_name}:")
                    print(f"  平均精确率: {avg_precision:.3f}")
                    print(f"  平均召回率: {avg_recall:.3f}")
                    print(f"  平均F1分数: {avg_f1:.3f}")
                    print(f"  F1≥0.8的设备数: {high_f1_devices}/{total_devices_with_algo} "
                          f"({high_f1_devices/total_devices_with_algo:.1%})")
        
        print("\n=== 指标说明 ===")
        print("• 精确率(Precision): 算法检测正确的比例，越高越好")
        print("• 召回率(Recall): 人工标记被正确检测的比例，越高越好")
        print("• F1分数: 精确率和召回率的调和平均，综合指标")
        print("• 漏检率(Miss Rate): 人工标记未被检测的比例，越低越好")
        print("• 误检率: 算法误报的比例，越低越好")
        print("• TP: 真正例，FP: 假正例，FN: 假负例")
        
        print("\n=== 匹配策略说明 ===")
        print("• 严格包含: 人工时间段完全包含算法时间段")
        print("• 时间重叠: 算法时间段与人工时间段有任意重叠")
        print(f"• IoU≥{IOU_THRESHOLD}: 时间重叠度（交并比）超过{IOU_THRESHOLD}")

if __name__ == "__main__":
    main()
