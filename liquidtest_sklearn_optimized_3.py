# -*- coding: utf-8 -*-
"""
...（上方注释同原来，可省略）...
新增：
  • 将 liquidLevelValue ≠ liquidLevel_clean 的“数据不一致”也作为 anomaly 段写入 labeled_anomalies.csv
"""

import os, glob
import numpy as np
import pandas as pd

# ====== 基本配置 & 阈值（与原版一致，略） ======
IN_DIR        = "label_data"
PATTERN       = "device_*.csv"
OUT_RAW_CSV   = "labeled_events_updated.csv"
OUT_FINAL_CSV = "labeled_events_final_12h.csv"
OUT_ANOM_CSV  = "labeled_anomalies.csv"

RISE_TH      = 0.8
DROP_TH      = 0.3
DRAIN_LEN_H  = 3
DROP_SUM_TH  = 2.0
FAST_WIN_H   = 5
FAST_NEED_H  = 4

SPIKE_TH     = 8.0
ANOM_WINDOW  = pd.Timedelta(hours=5)  # anomaly 结束后闭区间 [end, end+5h] 不判定
MAX_GAP_H    = 3

POST_12H_DROP_MM   = 2.5
POST_12H_FWD_TOL_M = 60

RAW_CLEAN_MISMATCH_WINDOW = pd.Timedelta(hours=1)  # 不一致后闭区间 [t, t+3h] 不判定

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

anom_records  = []
raw_records   = []
final_records = []

for fp in glob.glob(os.path.join(IN_DIR, PATTERN)):
    code = os.path.basename(fp).replace("device_","").replace(".csv","")
    df = pd.read_csv(fp, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    if "liquidLevel_clean" not in df.columns:
        print(f"[{code}] 缺少 'liquidLevel_clean'，跳过")
        continue

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
        anomaly_mask = g["is_night"] & (g["diff"] > RISE_TH)
        anomaly_segs = extract_segments_from_mask(g["msgTime"], anomaly_mask.values)
        anomaly_ends = [e for (_, e) in anomaly_segs]
        for s, e in anomaly_segs:
            anom_records.append({"code": code, "start": s, "end": e, "label": "anomaly"})
            raw_records.append({"code": code, "start": s, "end": e, "label": "anomaly"})

        # === 新增：数据不一致 anomaly（也写入 anomalies.csv）===
        mismatch_windows = []
        if "liquidLevelValue" in g.columns:
            # 直接不等；如需容差可用 np.isclose
            mismatch_mask = g["liquidLevelValue"] != g["liquidLevel_clean"]
            if mismatch_mask.any():
                # 1) 把不一致“点”合并为“段”，纳入 anomaly 输出
                mismatch_segs = extract_segments_from_mask(g["msgTime"], mismatch_mask.values)
                for s, e in mismatch_segs:
                    anom_records.append({"code": code, "start": s, "end": e, "label": "anomaly"})
                    raw_records.append({"code": code, "start": s, "end": e, "label": "anomaly"})
                # 2) 生成不一致后的“闭区间”隔离窗 [t, t+3h]，用于屏蔽 drain
                for t in g.loc[mismatch_mask, "msgTime"]:
                    mismatch_windows.append((t, t + RAW_CLEAN_MISMATCH_WINDOW))
        else:
            mismatch_mask = pd.Series(False, index=g.index)

        # === drain 候选（3h累计 OR 5h≥4h速降）===
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

            # anomaly 近邻隔离：闭区间 [ae, ae + ANOM_WINDOW]
            if any(ae <= s <= ae + ANOM_WINDOW for ae in anomaly_ends):
                continue

            # 不一致后的隔离：闭区间 [t, t+3h]
            if any(t0 <= s <= t1 for (t0, t1) in mismatch_windows):
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

# ====== 保存结果 ======
if anom_records:
    pd.DataFrame(anom_records).to_csv(OUT_ANOM_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {OUT_ANOM_CSV}（仅 anomaly，含夜间上升 & 数据不一致）")
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
