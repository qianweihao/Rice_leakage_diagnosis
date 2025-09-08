#!/usr/bin/env python3
# multiclass_multi_model_with_threshold.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# ====== 配置区域（请根据实际路径修改） ======
DATA_DIR               = "label_data"               # 存放 device_*.csv 的目录
EVENTS_FILE            = "labeled_events.csv"       # anomaly/drain 标注文件
TRAIN_DEVICE_COUNT     = 30                         # 用于训练的设备数
WINDOW_LEN             = 5                          # 计算 cum_drop 的滑动窗口
OUTPUT_METRICS_CSV     = "model_performance_threshold.csv"

# 短期调优参数
DRAIN_WEIGHT           = 10   # 对 drain 类加大权重
THRESH_DR              = 0.6  # drain 概率阈值

# 1. 读取标注
events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])

# 2. 列出所有设备编码
device_files = [f for f in os.listdir(DATA_DIR) if f.startswith("device_") and f.endswith(".csv")]
all_codes    = [f.replace("device_","").replace(".csv","") for f in device_files]

# 3. 随机划分训练/验证设备
train_codes, val_codes = train_test_split(
    all_codes,
    train_size=TRAIN_DEVICE_COUNT,
    random_state=42
)

# 4. 加载单设备数据并构造特征、标签
def load_device_data(code):
    df = pd.read_csv(os.path.join(DATA_DIR, f"device_{code}.csv"), parse_dates=["msgTime"])
    df = df.sort_values("msgTime").reset_index(drop=True)
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # 多分类标签：0=normal, 1=anomaly, 2=drain
    df["label"] = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        mask = (df["msgTime"]>=ev.start) & (df["msgTime"]<=ev.end)
        df.loc[mask, "label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        mask = (df["msgTime"]>=ev.start) & (df["msgTime"]<=ev.end)
        df.loc[mask, "label"] = 2
    df = df.dropna(subset=["diff","cum_drop"])
    X = df[["diff","cum_drop"]]
    y = df["label"].astype(int)
    return X, y

# 5. 构建训练集 & 验证集
def build_dataset(codes):
    Xs, ys = [], []
    for c in codes:
        Xc, yc = load_device_data(c)
        Xs.append(Xc); ys.append(yc)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

print(f"训练设备: {len(train_codes)}, 验证设备: {len(val_codes)}")
print(f"验证集分布:\n{y_val.value_counts()}")

# 6. 定义模型（给 drain 加权）
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight={0:1,1:1,2:DRAIN_WEIGHT},
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight={0:1,1:1,2:DRAIN_WEIGHT},
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 7. 训练、阈值判断 & 评估
results = []
classes = ["normal","anomaly","drain"]

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    base_pred = mdl.predict(X_val)
    # 如果支持概率输出，就用阈值提升 drain 精度
    if hasattr(mdl, "predict_proba"):
        probs = mdl.predict_proba(X_val)[:,2]
        y_pred = np.where(probs >= THRESH_DR, 2, base_pred)
    else:
        y_pred = base_pred

    print(f"\n=== {name} (drain threshold={THRESH_DR}) ===")
    print(classification_report(y_val, y_pred, target_names=classes, digits=4))

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_val, y_pred, labels=[0,1,2], zero_division=0
    )
    for cls, p, r, f, s in zip(classes, prec, rec, f1, sup):
        results.append({
            "Model":     name,
            "Class":     cls,
            "Precision": p,
            "Recall":    r,
            "F1-score":  f,
            "Support":   s,
            "Accuracy":  acc
        })

# 8. 保存结果
pd.DataFrame(results).to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存性能指标：{OUTPUT_METRICS_CSV}")
