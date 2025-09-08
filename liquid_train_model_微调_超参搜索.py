#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# 配置
DATA_DIR             = "label_data"
EVENTS_FILE          = "labeled_events.csv"
TRAIN_DEVICE_COUNT   = 30
WINDOW_LEN           = 5
DRAIN_WEIGHT         = 10
OUTPUT_THRESHOLD_CSV = "threshold_tuning_results.csv"
OUTPUT_MODEL_CSV     = "simple_tuning_performance.csv"

# 1. 读标注
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# 2. 收集设备编码并切分
codes = [f.replace("device_","").replace(".csv","") 
         for f in os.listdir(DATA_DIR) if f.startswith("device_")]
train_codes, val_codes = train_test_split(codes, train_size=TRAIN_DEVICE_COUNT, random_state=42)

# 3. 加载单设备特征和标签
def load_device(code):
    df = pd.read_csv(os.path.join(DATA_DIR, f"device_{code}.csv"), parse_dates=["msgTime"])
    df.sort_values("msgTime", inplace=True)
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    df["label"]    = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        df.loc[(df.msgTime>=ev.start)&(df.msgTime<=ev.end),"label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        df.loc[(df.msgTime>=ev.start)&(df.msgTime<=ev.end),"label"] = 2
    df.dropna(subset=["diff","cum_drop"], inplace=True)
    return df[["diff","cum_drop"]], df["label"]

def build(codes):
    Xs, ys = [], []
    for c in codes:
        Xc, yc = load_device(c)
        Xs.append(Xc); ys.append(yc)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

X_train, y_train = build(train_codes)
X_val,   y_val   = build(val_codes)

# 4. 超参搜索 (GridSearch + TimeSeriesSplit)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 3, 5]
}
tscv = TimeSeriesSplit(n_splits=3)
base = RandomForestClassifier(class_weight={0:1,1:1,2:DRAIN_WEIGHT}, random_state=42)
gs = GridSearchCV(base, param_grid, cv=tscv, scoring="f1_weighted", n_jobs=-1)
gs.fit(X_train, y_train)
best_rf = gs.best_estimator_

# 5. 概率校准
cal = CalibratedClassifierCV(best_rf, cv=3, method="isotonic")
cal.fit(X_train, y_train)

# 6. 阈值扫描
probs  = cal.predict_proba(X_val)[:,2]
y_true = (y_val == 2).astype(int)
thresholds = np.linspace(0,1,101)
records = []
for t in thresholds:
    pred = (probs >= t).astype(int)
    p,r,f,_ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    records.append({"threshold": t, "precision": p, "recall": r, "f1": f})
df_thr = pd.DataFrame(records)
df_thr.to_csv(OUTPUT_THRESHOLD_CSV, index=False)

best = df_thr.loc[df_thr.f1.idxmax()]
t0   = best.threshold
print("最佳阈值：", best.to_dict())

# 7. 最终评估
y_pred = np.where(probs >= t0, 2, cal.predict(X_val))
print(classification_report(y_val, y_pred, target_names=["normal","anomaly","drain"]))

perf = []
for name, clf in [("CalibratedRF", cal)]:
    prob_v = clf.predict_proba(X_val)[:,2]
    y_p    = np.where(prob_v >= t0, 2, clf.predict(X_val))
    acc    = accuracy_score(y_val, y_p)
    pr, rc, f1s, sup = precision_recall_fscore_support(y_val, y_p, labels=[0,1,2], zero_division=0)
    for cls, p_val, r_val, f_val, s_val in zip(["normal","anomaly","drain"], pr, rc, f1s, sup):
        perf.append([name, cls, p_val, r_val, f_val, s_val, acc])
pd.DataFrame(perf, columns=["Model","Class","Precision","Recall","F1","Support","Accuracy"])\
  .to_csv(OUTPUT_MODEL_CSV, index=False)
