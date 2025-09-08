#!/usr/bin/env python3
# multi_model_eval_save.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====== 配置 ======
DATA_DIR           = "label_data"               # 存放 device_*.csv 的目录
EVENTS_FILE        = "labeled_events.csv"       # 先前生成的标注文件
TRAIN_DEVICE_COUNT = 30                         # 30 台设备做训练
WINDOW_LEN         = 5                          # 计算 cum_drop 的滑动窗口
OUTPUT_METRICS_CSV = "model_performance.csv"    # 输出性能文件

# 1. 读取事件标注
events = pd.read_csv(EVENTS_FILE, parse_dates=["start", "end"])

# 2. 列出所有设备编码
device_files = [f for f in os.listdir(DATA_DIR) if f.startswith("device_") and f.endswith(".csv")]
all_codes    = [f.split("device_")[1].split(".csv")[0] for f in device_files]

# 3. 随机划分训练/验证设备
train_codes, val_codes = train_test_split(
    all_codes,
    train_size=TRAIN_DEVICE_COUNT,
    random_state=42
)

print(f"训练设备 ({len(train_codes)}): {train_codes}")
print(f"验证设备 ({len(val_codes)}): {val_codes}")

# 4. 加载单设备数据并构造特征与标签
def load_device_data(code):
    df = pd.read_csv(os.path.join(DATA_DIR, f"device_{code}.csv"), parse_dates=["msgTime"])
    df = df.sort_values("msgTime").reset_index(drop=True)
    # 特征
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # 标签：初始化为 0，drain 段内设为 1
    df["label"] = 0
    evs = events.query("code==@code and label=='drain'")
    for _, ev in evs.iterrows():
        mask = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[mask, "label"] = 1
    # 丢掉无法计算特征的行
    df = df.dropna(subset=["diff", "cum_drop"])
    return df[["diff", "cum_drop"]], df["label"].astype(int)

# 5. 根据设备列表构建数据集
def build_dataset(codes):
    X_parts, y_parts = [], []
    for c in codes:
        Xc, yc = load_device_data(c)
        X_parts.append(Xc)
        y_parts.append(yc)
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    return X, y

X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

print(f"训练样本: {len(y_train)}, 验证样本: {len(y_val)}")
print("验证集正/负样本分布：", y_val.value_counts().to_dict())

# 6. 定义要训练的模型
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 7. 训练、评估并收集性能
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    results.append({
        "Model":        name,
        "Accuracy":     accuracy_score(y_val, y_pred),
        "Precision(1)": precision_score(y_val, y_pred, zero_division=0),
        "Recall(1)":    recall_score(y_val, y_pred, zero_division=0),
        "F1-score(1)":  f1_score(y_val, y_pred, zero_division=0)
    })
    print(f"=== {name} ===")
    print(f"  Acc: {results[-1]['Accuracy']:.4f}  "
          f"P: {results[-1]['Precision(1)']:.4f}  "
          f"R: {results[-1]['Recall(1)']:.4f}  "
          f"F1: {results[-1]['F1-score(1)']:.4f}")

# 8. 保存所有模型性能到 CSV
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ 模型性能已保存到 {OUTPUT_METRICS_CSV}")
