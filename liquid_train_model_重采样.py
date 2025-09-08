#!/usr/bin/env python3
# multiclass_multi_model_with_manual_oversampling.py

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

# ====== 配置 ======
DATA_DIR            = "label_data"             # 存放 device_*.csv 的目录
EVENTS_FILE         = "labeled_events.csv"     # anomaly/drain 标注文件
TRAIN_DEVICE_COUNT  = 30                       # 用于训练的设备数量
WINDOW_LEN          = 5                        # 计算 cum_drop 的滑动窗口
OVERSAMPLE_TO       = 500                      # drain 类手动过采样到总数
DRAIN_WEIGHT        = 10                       # class_weight 中 drain 的权重
THRESH_DR           = 0.5                      # drain 概率阈值
OUTPUT_METRICS_CSV  = "model_performance_oversample.csv"

# 1. 读取标注
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# 2. 列出设备编码
device_files = [f for f in os.listdir(DATA_DIR)
                if f.startswith("device_") and f.endswith(".csv")]
all_codes = [f.replace("device_","").replace(".csv","") for f in device_files]

# 3. 分割训练/验证设备
train_codes, val_codes = train_test_split(
    all_codes,
    train_size=TRAIN_DEVICE_COUNT,
    random_state=42
)

print(f"Train devices ({len(train_codes)}): {train_codes}")
print(f"Val devices   ({len(val_codes)}): {val_codes}")

# 4. 加载单设备数据，构造特征和标签
def load_device_data(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    df["label"]    = 0
    # 标注 anomaly
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        m = (df.msgTime >= ev.start) & (df.msgTime <= ev.end)
        df.loc[m, "label"] = 1
    # 标注 drain
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        m = (df.msgTime >= ev.start) & (df.msgTime <= ev.end)
        df.loc[m, "label"] = 2
    return df.dropna(subset=["diff","cum_drop"])[["diff","cum_drop","label"]]

# 5. 构建训练 & 验证集
df_train = pd.concat([load_device_data(c) for c in train_codes], ignore_index=True)
df_val   = pd.concat([load_device_data(c) for c in val_codes],   ignore_index=True)

print("原始训练分布：")
print(df_train.label.value_counts())

# 6. 手动对 drain 类过采样到 OVERSAMPLE_TO 条
df_drain = df_train[df_train.label == 2]
n_current = len(df_drain)
if n_current < OVERSAMPLE_TO:
    n_extra = OVERSAMPLE_TO - n_current
    df_drain_extra = df_drain.sample(n=n_extra, replace=True, random_state=42)
    df_train = pd.concat([df_train, df_drain_extra], ignore_index=True)
    print(f"对 drain 过采样：从 {n_current} 增加到 {len(df_train[df_train.label==2])}")
else:
    print(f"drain 样本已 ≥ {OVERSAMPLE_TO}，无需过采样")

# 7. 拆分特征与标签
X_train = df_train[["diff","cum_drop"]]
y_train = df_train["label"]
X_val   = df_val[["diff","cum_drop"]]
y_val   = df_val["label"]

# 8. 定义模型列表并训练/评估
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

results = []
classes = ["normal","anomaly","drain"]

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    base_pred = mdl.predict(X_val)
    if hasattr(mdl, "predict_proba"):
        probs  = mdl.predict_proba(X_val)[:,2]
        # 仅当 drain 概率大于阈值才判为 drain
        y_pred = np.where(probs >= THRESH_DR, 2, base_pred)
    else:
        y_pred = base_pred

    print(f"\n=== {name} (oversampled drain + threshold {THRESH_DR}) ===")
    print(classification_report(y_val, y_pred, target_names=classes, digits=4))

    acc, (prec, rec, f1, sup) = (
        accuracy_score(y_val, y_pred),
        precision_recall_fscore_support(y_val, y_pred, labels=[0,1,2], zero_division=0)
    )
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

# 9. 保存评估结果
pd.DataFrame(results).to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 结果已保存到 {OUTPUT_METRICS_CSV}")
