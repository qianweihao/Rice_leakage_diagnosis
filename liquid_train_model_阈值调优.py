#!/usr/bin/env python3
# threshold_tuning.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# ====== 配置 ======
DATA_DIR           = "label_data"               # 存放 device_*.csv 的目录
EVENTS_FILE        = "labeled_events.csv"       # 标注文件
TRAIN_DEVICE_COUNT = 30                         # 用于训练的设备数
WINDOW_LEN         = 5                          # 计算 cum_drop 的滑动窗口
OUTPUT_CSV         = "threshold_tuning_results.csv"

# 可选：如果希望给 drain 类加权
DRAIN_WEIGHT       = 10                         # class_weight 中 drain 的权重

# ====== 1. 读取标注 ======
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# ====== 2. 收集设备编码 ======
device_files = [f for f in os.listdir(DATA_DIR)
                if f.startswith("device_") and f.endswith(".csv")]
all_codes = [f.replace("device_","").replace(".csv","") for f in device_files]

# ====== 3. 划分训练/验证设备 ======
train_codes, val_codes = train_test_split(
    all_codes,
    train_size=TRAIN_DEVICE_COUNT,
    random_state=42
)

# ====== 4. 加载数据函数 ======
def load_device_data(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # 多分类标签：0=normal,1=anomaly,2=drain
    df["label"] = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        mask = (df.msgTime >= ev.start) & (df.msgTime <= ev.end)
        df.loc[mask, "label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        mask = (df.msgTime >= ev.start) & (df.msgTime <= ev.end)
        df.loc[mask, "label"] = 2
    df = df.dropna(subset=["diff","cum_drop"])
    return df[["diff","cum_drop"]], df["label"].astype(int)

# ====== 5. 构建训练/验证集 ======
def build_dataset(codes):
    Xs, ys = [], []
    for c in codes:
        Xc, yc = load_device_data(c)
        Xs.append(Xc)
        ys.append(yc)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

# ====== 6. 训练一个示例模型（以 RandomForest 为例） ======
model = RandomForestClassifier(
    n_estimators=100,
    class_weight={0:1, 1:1, 2:DRAIN_WEIGHT},
    random_state=42
)
model.fit(X_train, y_train)

# ====== 7. 获取 drain 类概率 & 真值 ======
probs = model.predict_proba(X_val)[:, 2]      # p(label==2)
y_true = (y_val == 2).astype(int)            # 1 for drain, 0 otherwise

# ====== 8. 阈值扫描 ======
thresholds = np.linspace(0.0, 1.0, 101)
records = []
for t in thresholds:
    y_pred = (probs >= t).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    records.append({"threshold": t, "precision": p, "recall": r, "f1": f})

df_results = pd.DataFrame(records)

# ====== 9. 找到最佳阈值 ======
best = df_results.loc[df_results["f1"].idxmax()]
print("=== 最佳阈值 (按 F1 最大化) ===")
print(f"threshold = {best.threshold:.2f}")
print(f"precision = {best.precision:.2f}")
print(f"recall    = {best.recall:.2f}")
print(f"f1-score  = {best.f1:.2f}")

# ====== 10. 保存结果 ======
df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 阈值调优结果已保存到 {OUTPUT_CSV}")
