# -*- coding: utf-8 -*-
#主流程
"""
输出：
  1) labeled_events_updated.csv    (anomaly & drain,未叠加12h)
  2) labeled_events_final_12h.csv  （仅 drain,已通过12h≥2.5mm)
  3) labeled_anomalies.csv         （仅 anomaly:夜间上升 & 数据不一致）
"""

import os, glob
import numpy as np
import pandas as pd

# ====== 基本配置 ======
IN_DIR        = "label_data"
PATTERN       = "device_*.csv"
OUT_RAW_CSV   = "labeled_events_updated.csv"
OUT_FINAL_CSV = "labeled_events_final_12h.csv"
OUT_ANOM_CSV  = "labeled_anomalies.csv"

# ====== 阈值参数 ======
RISE_TH      = 0.8   # 夜间异常上升 diff 阈值（mm/h）
DROP_TH      = 0.3   # 速降阈值（mm/h）
DRAIN_LEN_H  = 3     # 3h 累计窗口
DROP_SUM_TH  = 2.0   # 3h 累计降幅阈值（mm）
FAST_WIN_H   = 5     # 多数小时窗口
FAST_NEED_H  = 4     # 窗口内至少 4 小时满足速降

SPIKE_TH     = 8.1   # 起始 vs 前5时段突变阈值（mm）
ANOM_WINDOW  = pd.Timedelta(hours=5)  # 异常结束后的闭区间隔离 [ae, ae+5h]
MAX_GAP_H    = 3     # >3h 断块

POST_12H_DROP_MM   = 2.5   # 12h 后置降幅阈值（mm）
POST_12H_FWD_TOL_M = 60    # 12h 取点向后容差（分钟）

# 数据不一致对称隔离窗（闭区间 [t-1h, t+1h]）
MISMATCH_HALF_WINDOW_H = 3

# ====== 工具函数 ======
def extract_segments_from_mask(times: pd.Series, mask: np.ndarray):
    """从布尔掩码中提取连续时间段"""
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
    """3 小时累计下降 < -th → 覆盖掩码（将该 3h 窗口三点全部置 True）"""
    roll = diff.rolling(window=window, min_periods=window).sum()
    mask = np.zeros(len(diff), dtype=bool)
    idxs = np.where(roll < -th)[0]
    for idx in idxs:
        s = max(0, idx - window + 1)
        mask[s:idx+1] = True
    return mask

def cover_by_5h4_fastdrops(diff: pd.Series, win_h: int, need_h: int, drop_th: float) -> np.ndarray:
    """任意 5 小时窗口内 ≥need_h 小时满足 diff < -drop_th → 覆盖掩码（对该窗口全置 True）"""
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
    """返回 df 中第一个时间 >= t 的液位值，若时间差 <= tol_minutes 则返回该值，否则返回 None"""
    sub = df[df["msgTime"] >= t]
    if sub.empty:
        return None
    dt = sub["msgTime"].iloc[0] - t
    if dt <= pd.Timedelta(minutes=tol_minutes):
        return float(sub["liquidLevel_clean"].iloc[0])
    return None

def overlaps(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    """闭区间的区间相交判定：[a0,a1] 与 [b0,b1] 是否有交集"""
    return (a0 <= b1) and (b0 <= a1)

# ====== 主流程 ======
anom_records  = []   # 仅 anomaly（夜间上升 & 数据不一致段）
raw_records   = []   # anomaly + drain（未做12h后置）
final_records = []   # 通过 12h≥2.5mm 后置确认的 drain

for fp in glob.glob(os.path.join(IN_DIR, PATTERN)):
    code = os.path.basename(fp).replace("device_","").replace(".csv","")
    df = pd.read_csv(fp, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    if "liquidLevel_clean" not in df.columns:
        print(f"[{code}] 缺少 'liquidLevel_clean'，跳过")
        continue

    # 分块（>3h 断开）
    df["delta_h"]  = df["msgTime"].diff().dt.total_seconds().div(3600.0)
    df["new_block"]= df["delta_h"].isna() | (df["delta_h"] > MAX_GAP_H)
    df["block_id"] = df["new_block"].cumsum()

    for _, g in df.groupby("block_id", sort=True):
        g = g.reset_index(drop=True).copy()
        if len(g) < 2:
            continue

        g["diff"] = g["liquidLevel_clean"].diff()
        hrs = g["msgTime"].dt.hour
        g["is_night"] = (hrs >= 22) | (hrs <= 5)

        # === 夜间上升 anomaly ===
        night_anom_mask = g["is_night"] & (g["diff"] > RISE_TH)
        night_anom_segs = extract_segments_from_mask(g["msgTime"], night_anom_mask.values)
        # 用于隔离的“异常结束时刻列表”
        anomaly_ends = [e for (_, e) in night_anom_segs]
        # 输出 anomaly（夜间上升）
        for s_a, e_a in night_anom_segs:
            anom_records.append({"code": code, "start": s_a, "end": e_a, "label": "anomaly"})
            raw_records.append({"code": code, "start": s_a, "end": e_a, "label": "anomaly"})

        # === 数据不一致 anomaly + 对称隔离窗 [t-1h, t+1h] ===
        mismatch_windows = []
        if "liquidLevelValue" in g.columns:
            # 如需容差，可改为：~np.isclose(value, clean, atol=0.01, rtol=0.0)
            mismatch_mask = g["liquidLevelValue"] != g["liquidLevel_clean"]
            if mismatch_mask.any():
                # 1) 不一致“段”写入 anomaly
                mismatch_segs = extract_segments_from_mask(g["msgTime"], mismatch_mask.values)
                for s_m, e_m in mismatch_segs:
                    anom_records.append({"code": code, "start": s_m, "end": e_m, "label": "anomaly"})
                    raw_records.append({"code": code, "start": s_m, "end": e_m, "label": "anomaly"})
                # 2) 生成对称隔离窗（闭区间）
                for t in g.loc[mismatch_mask, "msgTime"]:
                    mismatch_windows.append((t - pd.Timedelta(hours=MISMATCH_HALF_WINDOW_H),
                                             t + pd.Timedelta(hours=MISMATCH_HALF_WINDOW_H)))

        # === drain 候选（3h累计 OR 5h≥4h速降）===
        cover_cum3 = cover_by_cum3(g["diff"], window=DRAIN_LEN_H, th=DROP_SUM_TH)
        cover_5h4  = cover_by_5h4_fastdrops(g["diff"], win_h=FAST_WIN_H, need_h=FAST_NEED_H, drop_th=DROP_TH)
        cand_mask  = cover_cum3 | cover_5h4
        cand_segs  = extract_segments_from_mask(g["msgTime"], cand_mask)

        times = g["msgTime"].tolist()
        for s, e in cand_segs:
            # 位置索引（用于 Δ5 检查）
            try:
                pos = times.index(s)
            except ValueError:
                continue

            # 过滤①：起始 Δ5 突变（abs(lvl_start - lvl_prev5) > SPIKE_TH 则跳过）
            if pos >= 5:
                lvl_start = g.at[pos, "liquidLevel_clean"]
                lvl_prev5 = g.at[pos-5, "liquidLevel_clean"]
                if abs(lvl_start - lvl_prev5) > SPIKE_TH:
                    continue

            # 过滤②：异常近邻隔离（区间相交）
            # 屏蔽窗：对每个 anomaly_end = ae，闭区间 [ae, ae + ANOM_WINDOW]
            if any(overlaps(s, e, ae, ae + ANOM_WINDOW) for ae in anomaly_ends):
                continue

            # 过滤③：数据不一致对称隔离（区间相交）
            # 屏蔽窗：每个不一致时刻 t 的闭区间 [t-1h, t+1h]
            if any(overlaps(s, e, t0, t1) for (t0, t1) in mismatch_windows):
                continue

            # 通过过滤的 drain（未做12h后置）
            raw_records.append({"code": code, "start": s, "end": e, "label": "drain"})

            # === 12 小时后置确认：开始 +12h 下降 ≥ 2.5 mm（向后 ≤1h 取样）===
            # 起始水位
            exact = g[g["msgTime"] == s]
            if not exact.empty:
                level_s = float(exact["liquidLevel_clean"].iloc[0])
            else:
                prev = g[g["msgTime"] <= s].tail(1)
                if not prev.empty and (s - prev["msgTime"].iloc[0]) <= pd.Timedelta(hours=1):
                    level_s = float(prev["liquidLevel_clean"].iloc[0])
                else:
                    level_s = None

            # start+12h 水位（向后 ≤1h）
            t12 = s + pd.Timedelta(hours=12)
            level_12 = level_at_or_after(g, t12, tol_minutes=POST_12H_FWD_TOL_M)

            pass_12h = False
            delta_12 = None
            if (level_s is not None) and (level_12 is not None):
                delta_12 = level_12 - level_s  # 负值代表下降
                pass_12h = (delta_12 <= -POST_12H_DROP_MM)

            if pass_12h:
                final_records.append({
                    "code": code, "start": s, "end": e, "label": "drain",
                    "level_start_mm": level_s,
                    "level_at_12h_mm": level_12,
                    "delta_12h_mm": delta_12
                })

# ====== 保存结果 ======
if anom_records:
    pd.DataFrame(anom_records).to_csv(OUT_ANOM_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {OUT_ANOM_CSV}（仅 anomaly：夜间上升 & 数据不一致）")
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
