# -*- coding: utf-8 -*-
"""
Step: 根据"异常/排水"新逻辑提取事件段，并加入多重过滤 - 优化版本（已移除过滤②）

优化说明：
  • 降低检测阈值以捕获更多缓慢漏水事件
  • 缩短检测窗口长度
  • 放宽过滤条件
  • 针对9月11-12日和9月15-16日漏检问题进行优化

逻辑：
  • 异常 (anomaly)：夜间 (22:00–05:00) diff > RISE_TH
  • 排水 (drain)：要么连续 DRAIN_LEN_H 小时 diff < –DROP_TH
                要么连续 DRAIN_LEN_H 小时内累计下降 < –DROP_SUM_TH
  • 过滤①："起始点 vs 前5时段"突变过滤：若 abs(level_start – level_prev5) > SPIKE_TH 则忽略
  • 过滤②：（原“前1 vs 前5高点”过滤已移除）
  • 过滤②（重编号）：若 drain 段的 start 在任意 anomaly 段 end 后 ANOM_WINDOW 内，则忽略
  
输出：
  labeled_events_optimized.csv，字段：code,start,end,label
"""

import os, glob
import numpy as np
import pandas as pd

# ====== 优化后的配置 ======
IN_DIR       = "label_data"
PATTERN      = "device_*.csv"
OUTPUT_CSV   = "labeled_events_optimized.csv"  # 优化版本输出文件

# 优化后的阈值 - 降低检测门槛以捕获更多缓慢漏水事件
RISE_TH      = 0.8   # mm/h (原1.0，降低20%)
DROP_TH      = 0.3   # mm/h (原0.5，降低40%) - 更敏感的排水检测
DROP_SUM_TH  = 2.0   # mm (原4.0，降低50%) - 更低的累计下降要求
DRAIN_LEN_H  = 3     # 连续小时数窗长度 (原5，缩短到3小时)
SPIKE_TH     = 8.0   # mm，"起始点 vs 前5时段"突变阈值 (原5.0，放宽60%)
ANOM_WINDOW  = pd.Timedelta(hours=3)  # anomaly 后 X 小时内屏蔽 drain (原5→3小时)
MAX_GAP_H    = 3     # 时间连续性阈值（单位：小时），>3h 视为断点

def extract_segments_from_mask(times: pd.Series, mask: np.ndarray, min_len: int):
    """从布尔掩码中提取连续的时间段"""
    segments = []
    n = len(mask); i = 0
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            segments.append((times.iat[i], times.iat[j-1]))
        i = j
    return segments

records = []

for fp in glob.glob(os.path.join(IN_DIR, PATTERN)):
    code = os.path.basename(fp).replace("device_","").replace(".csv","")
    df = pd.read_csv(fp, parse_dates=["msgTime"])
    df = df.sort_values("msgTime").reset_index(drop=True)
    if "liquidLevel_clean" not in df.columns:
        print(f"[{code}] 缺少 'liquidLevel_clean'，跳过")
        continue

    # === 按时间差分块：相邻点相差 > MAX_GAP_H 小时就切断 ===
    df["delta_h"] = df["msgTime"].diff().dt.total_seconds().div(3600.0)
    # 首行也算新块；delta_h>MAX_GAP_H 视为断点
    df["new_block"] = df["delta_h"].isna() | (df["delta_h"] > MAX_GAP_H)
    df["block_id"] = df["new_block"].cumsum()

    # 逐块独立做检测，避免跨断点拼接
    for bid, g in df.groupby("block_id", sort=True):
        g = g.reset_index(drop=True)
        if len(g) == 0:
            continue

        # 1) 计算每小时差分(mm/h)（仅在当前块内）
        g["diff"] = g["liquidLevel_clean"].diff()

        # 2) 标记夜间
        hrs = g["msgTime"].dt.hour
        g["is_night"] = (hrs >= 22) | (hrs <= 5)

        # 3) 异常 (anomaly) 检测（块内）
        anomaly_mask = g["is_night"] & (g["diff"] > RISE_TH)
        anomaly_segs = extract_segments_from_mask(g["msgTime"], anomaly_mask.values, min_len=1)
        anomaly_ends = [end for (_, end) in anomaly_segs]
        for start, end in anomaly_segs:
            records.append({"code": code, "start": start, "end": end, "label": "anomaly"})

        # 4) Drain 候选段（块内）
        # 4.1 连续下降检测
        mask1 = g["diff"] < -DROP_TH
        segs1 = extract_segments_from_mask(g["msgTime"], mask1.values, min_len=DRAIN_LEN_H)

        # 4.2 累计下降检测
        g["cum_drop"] = g["diff"].rolling(window=DRAIN_LEN_H, min_periods=DRAIN_LEN_H).sum()
        segs2 = []
        idxs = np.where(g["cum_drop"] < -DROP_SUM_TH)[0]
        for idx in idxs:
            s = g["msgTime"].iat[idx - DRAIN_LEN_H + 1]
            e = g["msgTime"].iat[idx]
            segs2.append((s, e))

        candidates = segs1 + segs2

        # 5) 多重过滤（已移除原“前1 vs 前5高点过滤”）
        times = g["msgTime"].tolist()
        for start, end in candidates:
            try:
                pos = times.index(start)
            except ValueError:
                continue

            lvl_start = g.at[pos, "liquidLevel_clean"]

            # 过滤①：起始点 vs 前5时段突变（仍保留）
            if pos >= 5:
                lvl_prev5 = g.at[pos-5, "liquidLevel_clean"]
                if abs(lvl_start - lvl_prev5) > SPIKE_TH:
                    continue

            # （原过滤②：前1 vs 前5高点过滤 —— 已移除）

            # 过滤②（重编号）：与 anomaly 段的时间隔离
            skip = False
            for ae in anomaly_ends:
                if ae < start <= ae + ANOM_WINDOW:
                    skip = True
                    break
            if skip:
                continue

            # 通过所有过滤，记录 drain
            records.append({"code": code, "start": start, "end": end, "label": "drain"})


# 6) 保存结果
if records:
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成优化版本 {OUTPUT_CSV}，共 {len(out_df)} 条事件")
    print(f"📊 优化说明：")
    print(f"   - DROP_TH: 0.5 → 0.3 mm/h (降低40%)")
    print(f"   - DROP_SUM_TH: 4.0 → 2.0 mm (降低50%)")
    print(f"   - DRAIN_LEN_H: 5 → 3 小时 (缩短40%)")
    print(f"   - SPIKE_TH: 5.0 → 8.0 mm (放宽60%)")
    print(f"   - ANOM_WINDOW: 5 → 3 小时 (缩短40%)")
    print(f"   - 已移除过滤②：'前1时段 vs 前5时段'高点过滤")
else:
    print("⚠️ 未检测到任何事件，请检查阈值或输入数据。")
