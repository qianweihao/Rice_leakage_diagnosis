# -*- coding: utf-8 -*-
"""
Step: 根据最新"异常/排水"逻辑提取事件段 + 多重过滤 + 12小时后置确认
新增：输出单独的异常（anomaly）CSV

变更要点：
  • 用“任意 5 小时窗口内 ≥4 小时满足 diff < -0.3 mm/h”替代“连续 ≥3 小时速降”
  • 保留“3 小时累计下降 < -2.0 mm”的累计规则
  • 过滤：起始点 Δ5 突变过滤（8mm）、anomaly 邻近隔离（由 ANOM_WINDOW 定义，**闭区间**）
  • 后置确认：开始 +12h 水位下降 ≥ 2.5 mm（向后 ≤1h 容差）
输出：
  1) labeled_events_updated.csv    （anomaly & drain，未叠加12h）
  2) labeled_events_final_12h.csv  （仅 drain，已通过12h≥2.5mm）
  3) labeled_anomalies.csv         （仅 anomaly）
"""

import os, glob
import numpy as np
import pandas as pd

# ====== 基本配置 ======
IN_DIR        = "label_data"
PATTERN       = "device_*.csv"
OUT_RAW_CSV   = "labeled_events_updated.csv"     # 新规则+过滤（未做12h后置）
OUT_FINAL_CSV = "labeled_events_final_12h.csv"   # 新规则+过滤 + 12h后置
OUT_ANOM_CSV  = "labeled_anomalies.csv"          # 仅 anomaly 事件

# ====== 阈值参数 ======
# anomaly（夜间突升）
RISE_TH      = 0.8   # mm/h
# drain 检测（两条触发路满足任一即可成为候选）
DROP_TH      = 0.3   # mm/h，速降阈值
DRAIN_LEN_H  = 3     # 小时，用于 3h 累计规则滑窗长度
DROP_SUM_TH  = 2.0   # mm，3h 累计下降阈值（Σdiff < -2.0 mm）
FAST_WIN_H   = 5     # 小时；多数小时规则窗口长度
FAST_NEED_H  = 4     # 小时；在 FAST_WIN_H 内至少有 FAST_NEED_H 小时满足 diff < -DROP_TH

# 过滤
SPIKE_TH     = 8.0   # mm，起始点 vs 前5时段 突变过滤阈值

# === 关键修改：将 ANOM_WINDOW 定义为“anomaly_end 起，闭区间屏蔽”的时长 ===
# 解释：drain 段起点 s 若满足 ae <= s <= ae + ANOM_WINDOW（闭区间），则直接忽略；
#      必须到该区间结束后的下一个采样时刻才允许判定。
ANOM_WINDOW  = pd.Timedelta(hours=5)  # 例：5 小时 → 屏蔽 [anomaly_end, anomaly_end+5h]
MAX_GAP_H    = 3     # 小时，时间断点分块阈值

# 12 小时后置确认
POST_12H_DROP_MM   = 2.5     # mm，要求开始+12h 下降量 ≥ 2.5
POST_12H_FWD_TOL_M = 60      # 分钟，向后取点容差（若无点则视为失败）

# ====== 工具函数 ======
def extract_segments_from_mask(times: pd.Series, mask: np.ndarray):
    segs = []
    n = len(mask); i = 0
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < n and mask[j]:
            j += 1
        segs.append((times.iat[i], times.iat[j-1]))
        i = j
    return segs

def cover_by_cum3(diff: pd.Series, window: int, th: float) -> np.ndarray:
    roll = diff.rolling(window=window, min_periods=window).sum()
    mask = np.zeros(len(diff), dtype=bool)
    idxs = np.where(roll < -th)[0]
    for idx in idxs:
        s = max(0, idx - window + 1)
        mask[s:idx+1] = True
    return mask

def cover_by_5h4_fastdrops(diff: pd.Series, win_h: int, need_h: int, drop_th: float) -> np.ndarray:
    n = len(diff)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    fast = (diff < -drop_th).astype(int).values
    if n >= need_h:
        for i in range(0, n - win_h + 1):
            if fast[i:i+win_h].sum() >= need_h:
                mask[i:i+win_h] = True
    return mask

def level_at_or_after(df: pd.DataFrame, t: pd.Timestamp, tol_minutes: int):
    sub = df[df["msgTime"] >= t]
    if sub.empty:
        return None
    dt = sub["msgTime"].iloc[0] - t
    if dt <= pd.Timedelta(minutes=tol_minutes):
        return float(sub["liquidLevel_clean"].iloc[0])
    return None

# ====== 主流程 ======
anom_records  = []   # 仅 anomaly
raw_records   = []   # anomaly + drain（未做12h后置）
final_records = []   # 通过 12h≥2.5mm 后置确认的 drain

for fp in glob.glob(os.path.join(IN_DIR, PATTERN)):
    code = os.path.basename(fp).replace("device_","").replace(".csv","")
    df = pd.read_csv(fp, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    if "liquidLevel_clean" not in df.columns:
        print(f"[{code}] 缺少 'liquidLevel_clean'，跳过")
        continue

    # 时间分块
    df["delta_h"]  = df["msgTime"].diff().dt.total_seconds().div(3600.0)
    df["new_block"]= df["delta_h"].isna() | (df["delta_h"] > MAX_GAP_H)
    df["block_id"] = df["new_block"].cumsum()

    for bid, g in df.groupby("block_id", sort=True):
        g = g.reset_index(drop=True).copy()
        if len(g) < 2:
            continue

        # diff & 夜间
        g["diff"] = g["liquidLevel_clean"].diff()
        hrs = g["msgTime"].dt.hour
        g["is_night"] = (hrs >= 22) | (hrs <= 5)

        # anomaly 检测
        anomaly_mask = g["is_night"] & (g["diff"] > RISE_TH)
        anomaly_segs = extract_segments_from_mask(g["msgTime"], anomaly_mask.values)
        anomaly_ends = [e for (_, e) in anomaly_segs]
        for s, e in anomaly_segs:
            anom_records.append({"code": code, "start": s, "end": e, "label": "anomaly"})
            raw_records.append({"code": code, "start": s, "end": e, "label": "anomaly"})

        # drain 候选（3h 累计 OR 5h≥4h 速降）
        cover_cum3 = cover_by_cum3(g["diff"], window=DRAIN_LEN_H, th=DROP_SUM_TH)
        cover_5h4  = cover_by_5h4_fastdrops(g["diff"], win_h=FAST_WIN_H, need_h=FAST_NEED_H, drop_th=DROP_TH)
        cand_mask  = cover_cum3 | cover_5h4
        cand_segs  = extract_segments_from_mask(g["msgTime"], cand_mask)

        times = g["msgTime"].tolist()
        for s, e in cand_segs:
            try:
                pos = times.index(s)
            except ValueError:
                continue

            # 起始突变过滤
            if pos >= 5:
                lvl_start = g.at[pos, "liquidLevel_clean"]
                lvl_prev5 = g.at[pos-5, "liquidLevel_clean"]
                if abs(lvl_start - lvl_prev5) > SPIKE_TH:
                    continue

            # === 关键修改：anomaly 近邻隔离 —— 闭区间 [ae, ae + ANOM_WINDOW] ===
            # 例如：ae=01:00, ANOM_WINDOW=5h → 屏蔽 [01:00, 06:00]；07:00 才允许判断
            if any(ae <= s <= ae + ANOM_WINDOW for ae in anomaly_ends):
                continue

            # 通过过滤的 drain（未做12h后置）
            raw_records.append({"code": code, "start": s, "end": e, "label": "drain"})

            # 12 小时后置确认
            exact = g[g["msgTime"] == s]
            if not exact.empty:
                level_s = float(exact["liquidLevel_clean"].iloc[0])
            else:
                prev = g[g["msgTime"] <= s].tail(1)
                if not prev.empty and (s - prev["msgTime"].iloc[0]) <= pd.Timedelta(hours=1):
                    level_s = float(prev["liquidLevel_clean"].iloc[0])
                else:
                    level_s = None

            t12 = s + pd.Timedelta(hours=12)
            level_12 = level_at_or_after(g, t12, tol_minutes=POST_12H_FWD_TOL_M)

            pass_12h = False
            delta_12 = None
            if (level_s is not None) and (level_12 is not None):
                delta_12 = level_12 - level_s
                pass_12h = (delta_12 <= -POST_12H_DROP_MM)

            if pass_12h:
                final_records.append({
                    "code": code, "start": s, "end": e, "label": "drain",
                    "level_start_mm": level_s,
                    "level_at_12h_mm": level_12,
                    "delta_12h_mm": delta_12
                })

# 保存结果
if anom_records:
    pd.DataFrame(anom_records).to_csv(OUT_ANOM_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {OUT_ANOM_CSV}（仅 anomaly）")
else:
    print("⚠️ 未检测到 anomaly 事件。")

if raw_records:
    pd.DataFrame(raw_records).to_csv(OUT_RAW_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {OUT_RAW_CSV}（含 anomaly 与未叠加12h的 drain）")
else:
    print("⚠️ 未得到任何候选/过滤后的事件（raw）。")

if final_records:
    pd.DataFrame(final_records).to_csv(OUT_FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {OUT_FINAL_CSV}（通过 12h≥{POST_12H_DROP_MM}mm 后置确认的 drain）")
else:
    print("⚠️ 未有事件通过 12h 后置确认，请检查阈值或数据覆盖。")
