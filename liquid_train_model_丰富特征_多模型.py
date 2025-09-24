#!/usr/bin/env python3
# multiclass_multi_model_extended.py
#废弃模型-多模型效果较差
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# ===== Configuration =====
DATA_DIR           = "label_data"                       # 目录，存放 device_*.csv
EVENTS_FILE        = "labeled_events.csv"               # 标注文件
TRAIN_DEVICE_COUNT = 30                                 # 训练设备数
WINDOW_LEN         = 5                                  # 滑窗长度
OVERSAMPLE_TO      = 500                                # drain 类过采样到的样本数
DRAIN_WEIGHT       = 10                                 # class_weight 中 drain 的权重
THRESH_DR          = 0.43                               # drain 判定阈值
OUTPUT_CSV         = "model_performance_extended.csv"   # 输出文件

# 1. 读取标注
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# 2. 收集设备编码并划分训练/验证
codes = [
    f.replace("device_","").replace(".csv","")
    for f in os.listdir(DATA_DIR)
    if f.startswith("device_") and f.endswith(".csv")
]
train_codes, val_codes = train_test_split(codes, train_size=TRAIN_DEVICE_COUNT, random_state=42)

# 3. 特征加载函数（rich features + 滑窗统计）
def load_features(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    # 基础特征
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # 滚动统计：最大降幅 & 斜率方差
    for w in (3,5,12):
        df[f"max_drop_{w}h"]  = df["diff"].rolling(window=w, min_periods=1).min()
        df[f"var_slope_{w}h"] = df["diff"].rolling(window=w, min_periods=1).var()
    # 时间特征
    df["hour"]       = df["msgTime"].dt.hour
    df["is_night"]   = ((df["hour"]>=22)|(df["hour"]<=5)).astype(int)
    df["weekday"]    = df["msgTime"].dt.weekday
    df["is_weekend"] = (df["weekday"]>=5).astype(int)
    # 水位区间分桶
    df["level_bin"]  = pd.qcut(df["liquidLevel_clean"], q=4, labels=False, duplicates="drop")
    # 标签
    df["label"] = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        mask = (df["msgTime"]>=ev.start)&(df["msgTime"]<=ev.end)
        df.loc[mask, "label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        mask = (df["msgTime"]>=ev.start)&(df["msgTime"]<=ev.end)
        df.loc[mask, "label"] = 2
    df.dropna(subset=["diff","cum_drop","level_bin"], inplace=True)
    features = [
        "diff","cum_drop",
        "max_drop_3h","max_drop_5h","max_drop_12h",
        "var_slope_3h","var_slope_5h","var_slope_12h",
        "hour","is_night","is_weekend","level_bin"
    ]
    return df[features].astype(float), df["label"].astype(int)

def build_dataset(codes):
    Xs, ys = [], []
    for c in codes:
        Xc, yc = load_features(c)
        Xs.append(Xc); ys.append(yc)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

# 4. 构建训练 & 验证集
X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

# 5. 对 drain 类进行手动过采样
df_tr = pd.concat([X_train, y_train.rename("label")], axis=1)
drain_df = df_tr[df_tr.label==2]
if len(drain_df) < OVERSAMPLE_TO:
    extra = drain_df.sample(n=OVERSAMPLE_TO-len(drain_df), replace=True, random_state=42)
    df_tr = pd.concat([df_tr, extra], ignore_index=True)
X_train = df_tr.drop("label", axis=1)
y_train = df_tr["label"]

# 6. 定义要评估的模型列表
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000,
                                             class_weight={0:1,1:1,2:DRAIN_WEIGHT},
                                             random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100,
                                                 class_weight={0:1,1:1,2:DRAIN_WEIGHT},
                                                 random_state=42),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100,
                                                     random_state=42),
    "ExtraTrees":         ExtraTreesClassifier(n_estimators=100,
                                               class_weight={0:1,1:1,2:DRAIN_WEIGHT},
                                               random_state=42),
    "AdaBoost":           AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNeighbors":         KNeighborsClassifier(n_neighbors=5),
    "SVC":                SVC(probability=True,
                             class_weight={0:1,1:1,2:DRAIN_WEIGHT},
                             random_state=42),
    "MLP":                MLPClassifier(hidden_layer_sizes=(50,30),
                                        max_iter=500,
                                        random_state=42)
}

# 7. 训练、预测并评估
results = []
classes = ["normal","anomaly","drain"]
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    base_pred = mdl.predict(X_val)
    if hasattr(mdl, "predict_proba"):
        probs = mdl.predict_proba(X_val)[:,2]
        y_pred = np.where(probs >= THRESH_DR, 2, base_pred)
    else:
        y_pred = base_pred

    print(f"\n=== {name} ===")
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

# 8. 保存所有模型的性能对比
pd.DataFrame(results) \
  .to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ Extended performance saved to {OUTPUT_CSV}")
