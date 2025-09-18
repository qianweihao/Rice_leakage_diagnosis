# -*- coding: utf-8 -*-
"""
Step: 根据“异常/排水”新逻辑提取事件段，并加入多重过滤

逻辑：
  • 异常 (anomaly)：夜间 (22:00–05:00) diff > RISE_TH
  • 排水 (drain) ：要么连续 DRAIN_LEN_H 小时 diff < –DROP_TH
                要么连续 DRAIN_LEN_H 小时内累计下降 < –DROP_SUM_TH
  • 过滤①：“起始点 vs 前5时段”突变过滤：若 abs(level_start – level_prev5) > SPIKE_TH 则忽略
  • 过滤②：“前1时段 vs 前5时段”高点过滤：若 level_prev1 > level_start 且 level_prev1 > level_prev5 则忽略
  • 过滤③：若 drain 段的 start 在任意 anomaly 段 end 后 5 小时内，则忽略
  
输出：
  labeled_events.csv，字段：code,start,end,label
"""

import os, glob
import numpy as np
import pandas as pd

# ====== 配置 ======
IN_DIR       = "label_data"
PATTERN      = "device_*.csv"
OUTPUT_CSV   = "labeled_events.csv"

RISE_TH      = 1.0   # mm/h
DROP_TH      = 0.5   # mm/h
DROP_SUM_TH  = 4.0   # mm
DRAIN_LEN_H  = 5     # 连续小时数窗长度
SPIKE_TH     = 5.0   # mm，“起始点 vs 前5时段”突变阈值
ANOM_WINDOW  = pd.Timedelta(hours=5)  # anomaly 后 X 小时内屏蔽 drain
MAX_GAP_H    = 3     # 时间连续性阈值（单位：小时），>3h 视为断点

def extract_segments_from_mask(times: pd.Series, mask: np.ndarray, min_len: int):
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
        # 4.1 连续下降
        mask1 = g["diff"] < -DROP_TH
        segs1 = extract_segments_from_mask(g["msgTime"], mask1.values, min_len=DRAIN_LEN_H)

        # 4.2 累计下降（滚动窗口不会跨块）
        g["cum_drop"] = g["diff"].rolling(window=DRAIN_LEN_H, min_periods=DRAIN_LEN_H).sum()
        segs2 = []
        idxs = np.where(g["cum_drop"] < -DROP_SUM_TH)[0]
        for idx in idxs:
            s = g["msgTime"].iat[idx - DRAIN_LEN_H + 1]
            e = g["msgTime"].iat[idx]
            segs2.append((s, e))

        candidates = segs1 + segs2

        # 5) 多重过滤（全部基于块内相对位置，不会越界到断点外）
        times = g["msgTime"].tolist()
        for start, end in candidates:
            try:
                pos = times.index(start)
            except ValueError:
                continue

            lvl_start = g.at[pos, "liquidLevel_clean"]

            # 过滤①：起始点 vs 前5时段突变
            if pos >= 5:
                lvl_prev5 = g.at[pos-5, "liquidLevel_clean"]
                if abs(lvl_start - lvl_prev5) > SPIKE_TH:
                    continue

            # 过滤②：前1时段 vs 前5时段高点
            if pos >= 5:
                lvl_prev1 = g.at[pos-1, "liquidLevel_clean"]
                lvl_prev5 = g.at[pos-5, "liquidLevel_clean"]
                if lvl_prev1 > lvl_start and lvl_prev1 > lvl_prev5:
                    continue

            # 过滤③：与 anomaly 段的时间隔离（仅比较同块 anomaly）
            skip = False
            for ae in anomaly_ends:
                if ae < start <= ae + ANOM_WINDOW:
                    skip = True
                    break
            if skip:
                continue

            # 通过所有过滤，记录 drain（时间戳仍是全局时间）
            records.append({"code": code, "start": start, "end": end, "label": "drain"})


# 6) 保存结果
if records:
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {OUTPUT_CSV}，共 {len(out_df)} 条事件")
else:
    print("⚠️ 未检测到任何事件，请检查阈值或输入数据。")
