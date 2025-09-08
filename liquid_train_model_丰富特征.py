#!/usr/bin/env python3
# multiclass_multi_model_with_rich_features.py

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
DATA_DIR            = "label_data"              
EVENTS_FILE         = "labeled_events.csv"      
TRAIN_DEVICE_COUNT  = 30                        
WINDOW_LEN          = 5                         
OVERSAMPLE_TO       = 500                      
DRAIN_WEIGHT        = 10                        
THRESH_DR           = 0.50                      # 调优后最优阈值
OUTPUT_CSV          = "model_performance_rich_features.csv"

# ====== 1. 读取标注 ======
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# ====== 2. 收集设备编码 ======
device_files = [f for f in os.listdir(DATA_DIR) 
                if f.startswith("device_") and f.endswith(".csv")]
all_codes = [f.replace("device_","").replace(".csv","") for f in device_files]

# ====== 3. 划分训练/验证设备 ======
train_codes, val_codes = train_test_split(
    all_codes, train_size=TRAIN_DEVICE_COUNT, random_state=42
)

# ====== 4. 加载并构造丰富特征 ======
def load_device_data(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"])
    df = df.sort_values("msgTime").reset_index(drop=True)

    # 基础特征
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()

    # 时序窗口特征：过去 3h/5h/12h 最大降幅（最小 diff）、斜率方差
    for w in (3, 5, 12):
        df[f"max_drop_{w}h"] = df["diff"].rolling(window=w, min_periods=1).min()
        df[f"var_slope_{w}h"] = df["diff"].rolling(window=w, min_periods=1).var()

    # 时间特征
    df["hour"]       = df["msgTime"].dt.hour / 23.0       # 归一化 0–1
    df["is_night"]   = ((df["msgTime"].dt.hour >= 22) | 
                        (df["msgTime"].dt.hour <= 5)).astype(int)
    df["weekday"]    = df["msgTime"].dt.weekday           # 0=Mon...6=Sun
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # 水位绝对区间：按四分位数分桶
    df["level_bin"] = pd.qcut(df["liquidLevel_clean"],
                              q=4, labels=False)

    # 标签
    df["label"] = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        mask = (df.msgTime >= ev.start) & (df.msgTime <= ev.end)
        df.loc[mask, "label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        mask = (df.msgTime >= ev.start) & (df.msgTime <= ev.end)
        df.loc[mask, "label"] = 2

    # 丢弃 NaN
    df = df.dropna(subset=[
        "diff","cum_drop",
        "max_drop_3h","max_drop_5h","max_drop_12h",
        "var_slope_3h","var_slope_5h","var_slope_12h",
        "level_bin"
    ])

    # 特征列表
    features = [
        "diff", "cum_drop",
        "max_drop_3h","max_drop_5h","max_drop_12h",
        "var_slope_3h","var_slope_5h","var_slope_12h",
        "hour","is_night","is_weekend","level_bin"
    ]
    X = df[features].astype(float)
    y = df["label"].astype(int)
    return X, y

# ====== 5. 构建训练/验证集 ======
def build_dataset(codes):
    Xs, ys = [], []
    for c in codes:
        Xc, yc = load_device_data(c)
        Xs.append(Xc); ys.append(yc)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

print("训练样本分布：", y_train.value_counts())

# ====== 6. 对 drain 过采样 ======
df_tr = pd.concat([X_train, y_train.rename("label")], axis=1)
df_drain = df_tr[df_tr.label==2]
if len(df_drain) < OVERSAMPLE_TO:
    extra = df_drain.sample(n=OVERSAMPLE_TO-len(df_drain),
                            replace=True, random_state=42)
    df_tr = pd.concat([df_tr, extra], ignore_index=True)
X_train = df_tr.drop("label", axis=1)
y_train = df_tr["label"]

print("过采样后分布：", y_train.value_counts())

# ====== 7. 定义模型 & 训练评估 ======
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
        y_pred = np.where(probs >= THRESH_DR, 2, base_pred)
    else:
        y_pred = base_pred

    print(f"\n=== {name} ===")
    print(classification_report(y_val, y_pred, target_names=classes, digits=4))

    acc, (prec, rec, f1, sup) = (
        accuracy_score(y_val, y_pred),
        precision_recall_fscore_support(y_val, y_pred,
                                        labels=[0,1,2],
                                        zero_division=0)
    )
    for cls, p, r, f, s in zip(classes, prec, rec, f1, sup):
        results.append({
            "Model": name,
            "Class": cls,
            "Precision": p,
            "Recall":    r,
            "F1-score":  f,
            "Support":   s,
            "Accuracy":  acc
        })

# ====== 8. 保存结果 ======
pd.DataFrame(results)\
  .to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ 带丰富特征的评估结果已保存到 {OUTPUT_CSV}")
