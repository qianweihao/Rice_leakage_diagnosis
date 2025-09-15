# -*- coding: utf-8 -*-
"""
批量获取液位数据 → 清洗异常(>24) → 小时均值 → 去空小时 → 平滑
→ “滚动线性斜率 + 允许小间断合并”识别持续下降段 → 导出结果与图

输出目录：clean_results_smooth/
  device_{code}.csv                小时级数据 + 斜率/标记
  drain_segments_{code}.csv        持续下降段列表
  device_{code}.png                曲线图（红色高亮下降段）

文件名保持不变（只是放进目录里）。
"""

import os
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =================== 配置区 ===================
URL = "https://iland.zoomlion.com/open-sharing-platform/zlapi/irrigationApi/v1/getZnjsWaterHisByDeviceCode"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "dWCkcdbdSeMqHyMQmZruWzwHR30cspVH"
}

device_codes = ["157B025030016", "157B025030022", "157B025030001","157B025030032",
                "157B025030023","157B025010050","157B025010043","157B025030036",
                "157B025030026","157B025030033","157B025030025","157B025030037",
                "157B025030035","157B025030013","157B025030049","157B025030051",
                "157B025030047","157B025030034","157B025010025","157B025030003",
                "157B025010028","157B025030085","157B025030027","157B025030029",
                "157B025030039","157B025030076","157B025030068","157B025030077",
                "157B025030093","157B025030005","157B025030042","157B025010017",
                "157B025010006","157B025010054","157B025030044","157B025030031",
                "157B025030009","157B025030015","157B025030046","157B025030087",
                "157B025030008","157B025030079","157B025030014","157B025030091"]

START_DAY = "2025-06-15"
END_DAY   = "2025-09-14"

CHUNK_DAYS = 7
SLEEP_BETWEEN_CALLS = (1, 2)
SLEEP_BETWEEN_DEVICES = 5
MAX_RETRIES_PER_CHUNK = 4
THRESHOLD = 24                    # 异常上限
OUT_DIR = "clean_results_smooth"

# 持续下降检测参数
ROLL_WIN_H   = 6                  # 滚动线性回归窗口(小时)
SLOPE_TH     = -0.2               # 斜率阈值(mm/h)  小于此值视为下降
MAX_GAP      = 4                  # 允许的间断小时数
MIN_LEN      = 2                  # 最短持续小时
MIN_DROP     = 3                  # 累计降幅阈值(mm)

# =================== 初始化会话 ===================
os.makedirs(OUT_DIR, exist_ok=True)

session = requests.Session()
retries = Retry(total=3, backoff_factor=2,
                status_forcelist=[500, 502, 503, 504],
                raise_on_status=False)
adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
session.mount("http://", adapter)
session.mount("https://", adapter)

# =================== 工具函数 ===================

def daterange_chunks(start: datetime, end: datetime, step_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=step_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)

def parse_time_col(series: pd.Series) -> pd.Series:
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

def fetch_chunk(code: str, st: datetime, ed: datetime) -> pd.DataFrame:
    payload = {
        "deviceCode": code,
        "startDay": st.strftime("%Y-%m-%d"),
        "endDay":   ed.strftime("%Y-%m-%d")
    }
    for attempt in range(1, MAX_RETRIES_PER_CHUNK + 1):
        try:
            resp = session.post(URL, headers=HEADERS, json=payload, timeout=120, proxies={})
            if resp.status_code == 200:
                js = resp.json()
                data = js.get("data", [])
                return pd.DataFrame(data)
            else:
                print(f"[{code}] {st.date()}~{ed.date()} 状态码:{resp.status_code}, 重试{attempt}/{MAX_RETRIES_PER_CHUNK}")
        except requests.exceptions.RequestException as e:
            print(f"[{code}] {st.date()}~{ed.date()} 异常:{e}, 重试{attempt}/{MAX_RETRIES_PER_CHUNK}")
        time.sleep((2 ** attempt) + random.uniform(0, 1.5))
    print(f"[{code}] {st.date()}~{ed.date()} 多次重试失败，跳过该段")
    return pd.DataFrame()

def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "liquidLevelValue" not in df.columns:
        return df
    df["liquidLevelValue"] = pd.to_numeric(df["liquidLevelValue"], errors="coerce")
    df["is_outlier"] = df["liquidLevelValue"] > THRESHOLD
    df["liquidLevel_clean"] = df["liquidLevelValue"].where(~df["is_outlier"], pd.NA)

    if df["liquidLevel_clean"].isna().all():
        df["liquidLevel_clean"] = df["liquidLevelValue"]
        return df

    orig_index = df.index
    df = df.set_index("msgTime")
    try:
        df["liquidLevel_clean"] = (
            df["liquidLevel_clean"]
            .interpolate(method="time", limit=3)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
    except ValueError:
        df["liquidLevel_clean"] = (
            df["liquidLevel_clean"]
            .interpolate(method="linear", limit=3)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
    df = df.reset_index()
    df.index = orig_index
    return df

def hourly_average(df: pd.DataFrame) -> pd.DataFrame:
    df_hour = df.set_index("msgTime")
    agg = {}
    if "liquidLevelValue" in df_hour.columns:
        agg["liquidLevelValue"] = "mean"
    if "liquidLevel_clean" in df_hour.columns:
        agg["liquidLevel_clean"] = "mean"
    if "is_outlier" in df_hour.columns:
        agg["is_outlier"] = "max"

    df_hour = df_hour.resample("H").agg(agg)

    y_cols = [c for c in ["liquidLevel_clean", "liquidLevelValue"] if c in df_hour.columns]
    df_hour = df_hour.dropna(subset=y_cols, how="all")
    if "is_outlier" in df_hour.columns:
        df_hour["is_outlier"] = df_hour["is_outlier"].fillna(False)

    df_hour = df_hour.reset_index()

    for col in ["liquidLevelValue", "liquidLevel_clean"]:
        if col in df_hour.columns:
            df_hour[col] = df_hour[col].round(3)
    return df_hour

# -------- 平滑 --------
def hampel_filter(series, window_size=5, n_sigmas=3):
    s = series.copy()
    L = 1.4826
    med = s.rolling(window_size, center=True).median()
    diff = (s - med).abs()
    mad = diff.rolling(window_size, center=True).median()
    thr = n_sigmas * L * mad
    out = diff > thr
    s[out] = med[out]
    return s

# ======== 持续下降检测（滚动线性斜率）========
def rolling_lin_slope(y: np.ndarray, t_sec: np.ndarray, win: int = 6):
    n = len(y)
    res = np.full(n, np.nan)
    if win < 2:
        return res
    for i in range(win-1, n):
        xs = t_sec[i-win+1:i+1]
        ys = y[i-win+1:i+1]
        # 简单一次线性回归
        a, b = np.polyfit(xs, ys, 1)
        res[i] = a * 3600  # 秒→小时
    return res

def build_drop_mask(df, slope_col='roll_slope', slope_th=SLOPE_TH):
    return (df[slope_col] < slope_th) & df[slope_col].notna()

def merge_gaps(mask: pd.Series, max_gap: int = MAX_GAP) -> pd.Series:
    arr = mask.astype(int).values.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 1:
            i += 1
            continue
        j = i
        while j < n and arr[j] == 0:
            j += 1
        gap_len = j - i
        left_true  = (i - 1 >= 0 and arr[i - 1] == 1)
        right_true = (j < n and arr[j] == 1)
        if left_true and right_true and gap_len <= max_gap:
            arr[i:j] = 1
        i = j
    return pd.Series(arr.astype(bool), index=mask.index)

def extract_sustained_drop(df, mask_col='mask_final', val_col='liquidLevel_clean_smooth',
                           min_len=MIN_LEN, min_drop=MIN_DROP):
    segs = []
    m = df[mask_col].values
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        g = df.iloc[i:j]
        dur = len(g)
        drop = g[val_col].iloc[0] - g[val_col].iloc[-1]
        if dur >= min_len and drop >= min_drop:
            segs.append({
                'start': g['msgTime'].iloc[0],
                'end':   g['msgTime'].iloc[-1],
                'n_hours': dur,
                'drop_mm': round(drop, 3),
                'slope_median': round(g['roll_slope'].median(), 3)
            })
        i = j

    seg_df = pd.DataFrame(segs)
    df['is_drain'] = False
    for _, r in seg_df.iterrows():
        df.loc[(df['msgTime'] >= r['start']) & (df['msgTime'] <= r['end']), 'is_drain'] = True
    return df, seg_df

# ---------- 绘图 ----------
def plot_device_with_drain(df_flag, seg_df, code, y_col='liquidLevel_clean_smooth'):
    if y_col not in df_flag.columns or df_flag.empty:
        print(f"[{code}] 无可用液位数据列，跳过绘图")
        return

    df_plot = df_flag[df_flag[y_col].notna()]
    if df_plot.empty:
        print(f"[{code}] 绘图数据为空")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['msgTime'], df_plot[y_col], linewidth=1, label='level')

    # 高亮下降段
    if not seg_df.empty:
        for _, row in seg_df.iterrows():
            plt.axvspan(row['start'], row['end'], color='red', alpha=0.15)

    plt.xlabel("msgTime")
    plt.ylabel(y_col)
    plt.title(f"device{code} liquidLevel (Hourly Mean, smooth)")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, f"device_{code}.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

# =================== 主流程 ===================

def process_device(code: str, start_dt: datetime, end_dt: datetime):
    # 拉取
    dfs = []
    for st, ed in daterange_chunks(start_dt, end_dt, CHUNK_DAYS):
        df_part = fetch_chunk(code, st, ed)
        if not df_part.empty:
            dfs.append(df_part)
        time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))
    if not dfs:
        print(f"[{code}] 全区间无数据")
        return

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # 时间列
    time_col = None
    for col in ["msgTimeStr", "msgTime", "time"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        print(f"[{code}] 没有时间列: {df.columns.tolist()}")
        return

    df["msgTime"] = parse_time_col(df[time_col])
    df = df.dropna(subset=["msgTime"])
    df.sort_values("msgTime", inplace=True)
    df = df[(df["msgTime"] >= start_dt) & (df["msgTime"] <= end_dt)]
    if df.empty:
        print(f"[{code}] 清洗后无数据")
        return

    # 异常值处理
    df = clean_outliers(df)

    # 小时平均
    df_hour = hourly_average(df)
    if df_hour.empty:
        print(f"[{code}] 小时聚合后无数据")
        return

    # 平滑
    df_hour['liquidLevel_clean_smooth'] = hampel_filter(df_hour['liquidLevel_clean'], window_size=5, n_sigmas=3)

    # 滚动线性斜率
    ts_sec = df_hour['msgTime'].astype('int64') // 10**9
    y      = df_hour['liquidLevel_clean_smooth'].values
    df_hour['roll_slope'] = rolling_lin_slope(y, ts_sec.values, win=ROLL_WIN_H)

    # 掩码 + 合并
    mask_raw         = build_drop_mask(df_hour, slope_col='roll_slope', slope_th=SLOPE_TH)
    df_hour['mask_final'] = merge_gaps(mask_raw, max_gap=MAX_GAP)

    # 段提取
    df_flag, seg_df = extract_sustained_drop(df_hour,
                                             mask_col='mask_final',
                                             val_col='liquidLevel_clean_smooth',
                                             min_len=MIN_LEN,
                                             min_drop=MIN_DROP)

    # 保存
    csv_path = os.path.join(OUT_DIR, f"device_{code}.csv")
    df_flag.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[{code}] 保存CSV({len(df_flag)}条): {df_flag['msgTime'].min()} ~ {df_flag['msgTime'].max()} -> {csv_path}")

    drain_path = os.path.join(OUT_DIR, f"drain_segments_{code}.csv")
    seg_df.to_csv(drain_path, index=False, encoding="utf-8-sig")

    # 绘图
    plot_device_with_drain(df_flag, seg_df, code)

def main():
    start_dt = pd.to_datetime(START_DAY)
    end_dt   = pd.to_datetime(END_DAY)

    for idx, code in enumerate(device_codes, 1):
        print(f"\n===== ({idx}/{len(device_codes)}) 处理设备: {code} =====")
        process_device(code, start_dt, end_dt)
        time.sleep(SLEEP_BETWEEN_DEVICES)

    print("\n全部设备处理完毕。")

if __name__ == "__main__":
    main()
