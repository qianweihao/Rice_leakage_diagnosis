#!/usr/bin/env python3
# multiclass_multi_model_eval.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# ====== 配置 ======
DATA_DIR           = "label_data"               # 存放 device_*.csv 的目录
EVENTS_FILE        = "labeled_events.csv"       # 先前生成的标注文件
TRAIN_DEVICE_COUNT = 30                         # 30 台设备做训练
WINDOW_LEN         = 5                          # 计算 cum_drop 的滑动窗口
OUTPUT_METRICS_CSV = "model_performance.csv"    # 输出性能文件

# 1. 读取 anomaly/drain 标注
events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])

# 2. 列出所有设备编码
device_files = [
    f for f in os.listdir(DATA_DIR)
    if f.startswith("device_") and f.endswith(".csv")
]
all_codes = [f.replace("device_", "").replace(".csv", "") for f in device_files]

# 3. 随机划分训练/验证设备
train_codes, val_codes = train_test_split(
    all_codes,
    train_size=TRAIN_DEVICE_COUNT,
    random_state=42
)
print(f"Train devices ({len(train_codes)}): {train_codes}")
print(f"Val devices   ({len(val_codes)}): {val_codes}")

# 4. 加载单设备数据并构造特征与多分类标签
def load_device_data(code):
    df = pd.read_csv(
        os.path.join(DATA_DIR, f"device_{code}.csv"),
        parse_dates=["msgTime"]
    ).sort_values("msgTime").reset_index(drop=True)
    # 特征：单小时差分 & 窗口累计差分
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # 多分类标签：0=normal, 1=anomaly, 2=drain
    df["label"] = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        m = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[m, "label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        m = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[m, "label"] = 2
    df = df.dropna(subset=["diff", "cum_drop"])
    X = df[["diff", "cum_drop"]]
    y = df["label"].astype(int)
    return X, y

# 5. 构建训练/验证集
def build_dataset(codes):
    X_parts, y_parts = [], []
    for c in codes:
        Xc, yc = load_device_data(c)
        X_parts.append(Xc)
        y_parts.append(yc)
    return pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)

X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
print("Val distribution:\n", y_val.value_counts())

# 6. 定义多模型列表
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}

# 7. 多模型训练、评估并保存结果
results = []
classes = ["normal", "anomaly", "drain"]
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_val, y_pred, labels=[0,1,2], zero_division=0
    )
    print(f"\n=== {name} ===")
    print(classification_report(y_val, y_pred, target_names=classes, digits=4))
    for cls, p, r, f, s in zip(classes, prec, rec, f1, sup):
        results.append({
            "Model":    name,
            "Class":    cls,
            "Precision":p,
            "Recall":   r,
            "F1-score": f,
            "Support":  s,
            "Accuracy": acc
        })

# 8. 保存所有模型多分类性能到 CSV
df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Saved multi-model multiclass performance to {OUTPUT_METRICS_CSV}")
