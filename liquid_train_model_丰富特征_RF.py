#!/usr/bin/env python3
# rf_hyperparam_tuning.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, precision_recall_fscore_support
)

# ===== Configuration =====
DATA_DIR           = "label_data"
EVENTS_FILE        = "labeled_events.csv"
TRAIN_DEVICE_COUNT = 30
WINDOW_LEN         = 5
OVERSAMPLE_TO      = 500
DRAIN_WEIGHT       = 10
THRESH_RANGE       = np.linspace(0.0, 1.0, 101)
OUTPUT_GRID_CSV    = "rf_grid_search_results.csv"
OUTPUT_THRESH_CSV  = "rf_threshold_scan.csv"

# 1. Load annotations
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# 2. Gather device codes and split
codes = [
    f.replace("device_","").replace(".csv","")
    for f in os.listdir(DATA_DIR)
    if f.startswith("device_") and f.endswith(".csv")
]
train_codes, val_codes = train_test_split(codes, train_size=TRAIN_DEVICE_COUNT, random_state=42)

# 3. Feature loading (rich features)
def load_features(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    for w in (3,5,12):
        df[f"max_drop_{w}h"]  = df["diff"].rolling(window=w, min_periods=1).min()
        df[f"var_slope_{w}h"] = df["diff"].rolling(window=w, min_periods=1).var()
    df["hour"]       = df["msgTime"].dt.hour
    df["is_night"]   = ((df["hour"]>=22)|(df["hour"]<=5)).astype(int)
    df["weekday"]    = df["msgTime"].dt.weekday
    df["is_weekend"] = (df["weekday"]>=5).astype(int)
    df["level_bin"]  = pd.qcut(df["liquidLevel_clean"], q=4, labels=False, duplicates="drop")
    df["label"]      = 0
    for _,ev in events.query("code==@code and label=='anomaly'").iterrows():
        df.loc[(df["msgTime"]>=ev.start)&(df["msgTime"]<=ev.end),"label"]=1
    for _,ev in events.query("code==@code and label=='drain'").iterrows():
        df.loc[(df["msgTime"]>=ev.start)&(df["msgTime"]<=ev.end),"label"]=2
    df.dropna(subset=["diff","cum_drop","level_bin"], inplace=True)
    feats = [
        "diff","cum_drop",
        "max_drop_3h","max_drop_5h","max_drop_12h",
        "var_slope_3h","var_slope_5h","var_slope_12h",
        "hour","is_night","is_weekend","level_bin"
    ]
    return df[feats].astype(float), df["label"].astype(int)

def build_dataset(code_list):
    Xs, ys = [], []
    for c in code_list:
        Xc, yc = load_features(c)
        Xs.append(Xc); ys.append(yc)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)

# Build train/validation sets
X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

# Oversample drain
df_tr = pd.concat([X_train, y_train.rename("label")], axis=1)
drain_df = df_tr[df_tr.label==2]
if len(drain_df) < OVERSAMPLE_TO:
    extra = drain_df.sample(n=OVERSAMPLE_TO - len(drain_df), replace=True, random_state=42)
    df_tr = pd.concat([df_tr, extra], ignore_index=True)
X_train = df_tr.drop("label", axis=1)
y_train = df_tr["label"]

# 4. Grid search RPC hyperparameters
param_grid = {
    "n_estimators": [100,200,300],
    "max_depth": [5,10,20,None],
    "min_samples_leaf": [1,3,5],
    "max_features": ["sqrt","log2", None]
}
tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestClassifier(class_weight={0:1,1:1,2:DRAIN_WEIGHT}, random_state=42)
gs = GridSearchCV(rf, param_grid, cv=tscv, scoring="f1_weighted", n_jobs=-1)
gs.fit(X_train, y_train)

# Save grid search results
df_gs = pd.DataFrame(gs.cv_results_)[[
    "param_n_estimators","param_max_depth","param_min_samples_leaf","param_max_features",
    "mean_test_score","std_test_score"
]]
df_gs.to_csv(OUTPUT_GRID_CSV, index=False)
print("Best RF params:", gs.best_params_)

best_rf = gs.best_estimator_

# 5. Calibrate probabilities
cal_rf = CalibratedClassifierCV(best_rf, cv=3, method="sigmoid")
cal_rf.fit(X_train, y_train)

# 6. Threshold tuning
probs = cal_rf.predict_proba(X_val)[:,2]
y_true = (y_val==2).astype(int)
scan = []
for t in THRESH_RANGE:
    pred = (probs>=t).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(   y_true, pred, zero_division=0)
    f = f1_score(       y_true, pred, zero_division=0)
    scan.append({"threshold":t,"precision":p,"recall":r,"f1":f})
pd.DataFrame(scan).to_csv(OUTPUT_THRESH_CSV, index=False)
best = max(scan, key=lambda x: x["f1"])
print("Best threshold:", best)

# 7. Final evaluation
y_pred = np.where(probs>=best["threshold"], 2, best_rf.predict(X_val))
print(classification_report(y_val, y_pred, target_names=["normal","anomaly","drain"]))
