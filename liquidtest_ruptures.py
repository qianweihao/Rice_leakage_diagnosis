# -*- coding: utf-8 -*-
"""
批量获取液位数据 → 清洗异常(>24) → 小时均值 → 去空小时 → 平滑
→ 变化点检测 + 自适应阈值识别持续下降段 → 导出结果与图

输出目录:clean_results_cpd/
  device_{code}.csv           小时级数据 + is_drain 标记
  drain_segments_{code}.csv   持续下降段列表
  device_{code}.png           折线图（红色高亮下降段）
"""

import os
import time
import random
import requests
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ========== 配置区 ==========
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
END_DAY   = "2025-07-21"

CHUNK_DAYS = 7
SLEEP_BETWEEN_CALLS = (1,2)
SLEEP_BETWEEN_DEVICES = 5
MAX_RETRIES_PER_CHUNK = 4
THRESHOLD = 24
OUT_DIR = "clean_results_ruptures"

# 变化点检测与自适应阈值参数
CPD_MODEL = "l2"       # "l2" 或 "rbf"
CPD_PEN   = 10         # penalty，调越大段越少
# 分位自适应分位数
SLOPE_Q   = 0.10
DROP_Q    = 0.20
DUR_Q     = 0.20
# ===========================

os.makedirs(OUT_DIR, exist_ok=True)

# ========== 会话初始化 ==========
session = requests.Session()
retries = Retry(total=3, backoff_factor=2,
                status_forcelist=[500,502,503,504],
                raise_on_status=False)
adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ========== 工具函数 ==========
def daterange_chunks(start, end, step_days):
    cur = start
    while cur <= end:
        ce = min(cur + timedelta(days=step_days-1), end)
        yield cur, ce
        cur = ce + timedelta(days=1)

def parse_time_col(col):
    def _p(v):
        try:
            if isinstance(v,(int,float)):
                s=str(int(v))
            else:
                s=str(v)
            if s.isdigit():
                if len(s)==13: return pd.to_datetime(int(s),unit="ms")
                if len(s)==10: return pd.to_datetime(int(s),unit="s")
            return pd.to_datetime(s)
        except:
            return pd.NaT
    return col.map(_p)

def fetch_chunk(code, st, ed):
    payload = {"deviceCode":code,
               "startDay":st.strftime("%Y-%m-%d"),
               "endDay":ed.strftime("%Y-%m-%d")}
    for i in range(1, MAX_RETRIES_PER_CHUNK+1):
        try:
            r = session.post(URL, json=payload, headers=HEADERS, timeout=120)
            if r.status_code==200:
                return pd.DataFrame(r.json().get("data",[]))
            else:
                print(f"{code} {st}~{ed} 状态:{r.status_code} 重试{i}")
        except Exception as e:
            print(f"{code} {st}~{ed} 异常:{e} 重试{i}")
        time.sleep((2**i)+random.random())
    return pd.DataFrame()

def clean_outliers(df):
    if "liquidLevelValue" not in df: return df
    df["liquidLevelValue"]=pd.to_numeric(df["liquidLevelValue"],errors="coerce")
    df["is_outlier"]=df["liquidLevelValue"]>THRESHOLD
    df["liquidLevel_clean"]=df["liquidLevelValue"].mask(df["is_outlier"],pd.NA)
    if df["liquidLevel_clean"].isna().all():
        df["liquidLevel_clean"]=df["liquidLevelValue"]
        return df
    df=df.set_index("msgTime")
    df["liquidLevel_clean"]=(
        df["liquidLevel_clean"]
        .interpolate(method="time",limit=3)
        .ffill().bfill()
    )
    return df.reset_index()

def hourly_average(df):
    dfh = df.set_index("msgTime").resample("H").agg({
        "liquidLevelValue":"mean",
        "liquidLevel_clean":"mean",
        "is_outlier":"max"
    })
    dfh=dfh.dropna(subset=["liquidLevelValue","liquidLevel_clean"],how="all")
    dfh["is_outlier"]=dfh["is_outlier"].fillna(False)
    dfh=dfh.reset_index()
    dfh[["liquidLevelValue","liquidLevel_clean"]]=dfh[["liquidLevelValue","liquidLevel_clean"]].round(3)
    return dfh

def detect_drain_via_cpd(dfh):
    # 1. 变化点检测
    y = dfh["liquidLevel_clean"].values
    algo = rpt.Pelt(model=CPD_MODEL).fit(y)
    bkps = algo.predict(pen=CPD_PEN)
    # 2. 段统计
    segs=[]
    start=0
    for end in bkps:
        seg=dfh.iloc[start:end]
        if len(seg)>=3:
            x = seg["msgTime"].astype("int64")//10**9
            v = seg["liquidLevel_clean"].values
            a,b = np.polyfit(x,v,1)
            slope_h = a*3600
            drop   = v[0]-v[-1]
            dur    = len(seg)
            segs.append({
                "start":seg["msgTime"].iat[0],
                "end":seg["msgTime"].iat[-1],
                "n_hours":dur,
                "slope_h":slope_h,
                "drop_mm":drop
            })
        start=end
    segdf = pd.DataFrame(segs)
    if segdf.empty:
        dfh["is_drain"]=False
        return dfh, segdf
    # 3. 自适应阈值
    neg = segdf.loc[segdf["slope_h"]<0, "slope_h"]
    slope_th = neg.quantile(SLOPE_Q) if len(neg)>0 else -0.3
    drop_th  = segdf["drop_mm"].quantile(DROP_Q)
    dur_th   = segdf["n_hours"].quantile(DUR_Q)
    cond = (
        (segdf["slope_h"]<slope_th)&
        (segdf["drop_mm"]>=drop_th)&
        (segdf["n_hours"]>=dur_th)
    )
    segdf["is_drain"]=cond
    # 4. 回写标记
    dfh["is_drain"]=False
    for _,r in segdf[segdf["is_drain"]].iterrows():
        dfh.loc[
            (dfh["msgTime"]>=r["start"])&(dfh["msgTime"]<=r["end"]),
            "is_drain"
        ]=True
    return dfh, segdf

def plot_with_drain(dfh, segdf, code):
    plt.figure(figsize=(12,6))
    plt.plot(dfh["msgTime"], dfh["liquidLevel_clean"], label="level")
    for _,r in segdf[segdf["is_drain"]].iterrows():
        plt.axvspan(r["start"], r["end"], color="red", alpha=0.15)
    plt.xlabel("msgTime"); plt.ylabel("level (clean)")
    plt.title(f"device{code} drain detection")
    ax=plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f"device_{code}.png"),dpi=200)
    plt.close()

# ========== 主流程 ==========
def process_device(code):
    # 拉取 & 合并
    parts=[]
    for st,ed in daterange_chunks(
        pd.to_datetime(START_DAY), pd.to_datetime(END_DAY), CHUNK_DAYS
    ):
        dfp = fetch_chunk(code,st,ed)
        if not dfp.empty:
            parts.append(dfp)
        time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))
    if not parts:
        print(f"{code} 无数据")
        return
    df = pd.concat(parts,ignore_index=True).drop_duplicates()
    # 时间
    for col in ["msgTimeStr","msgTime","time"]:
        if col in df: df["msgTime"]=parse_time_col(df[col]); break
    df=df.dropna(subset=["msgTime"]).sort_values("msgTime")
    df=df[(df["msgTime"]>=START_DAY)&(df["msgTime"]<=END_DAY)]
    # 异常 & 小时均值
    df = clean_outliers(df)
    dfh= hourly_average(df)
    # 检测
    dfh, segdf = detect_drain_via_cpd(dfh)
    # 保存
    dfh.to_csv(os.path.join(OUT_DIR,f"device_{code}.csv"),index=False,encoding="utf-8-sig")
    segdf.to_csv(os.path.join(OUT_DIR,f"drain_segments_{code}.csv"),
                 index=False,encoding="utf-8-sig")
    # 画图
    plot_with_drain(dfh,segdf,code)
    print(f"{code} 完成: {len(dfh)} pts, drain segs={segdf['is_drain'].sum()}")

def main():
    for idx,code in enumerate(device_codes,1):
        print(f"[{idx}/{len(device_codes)}] {code}")
        process_device(code)
        time.sleep(SLEEP_BETWEEN_DEVICES)
    print("全部完成")

if __name__=="__main__":
    main()
