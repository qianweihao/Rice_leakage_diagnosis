#!/usr/bin/env python3
# advanced_pipeline_ensemble.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# ===== Configuration =====
DATA_DIR           = "label_data"            # directory containing device_*.csv
EVENTS_FILE        = "labeled_events.csv"    # anomaly/drain annotations
TRAIN_DEVICE_COUNT = 30
WINDOW_LEN         = 5

# Class weights for cost-sensitive learning
HGB_WEIGHT         = {0:1, 1:1, 2:20}
SGD_WEIGHT         = {0:1, 1:1, 2:20}

# Ensemble weights (how to combine probabilities)
HGB_WEIGHT_ENS     = 0.6
SGD_WEIGHT_ENS     = 0.4

# Range of thresholds to scan for best F1 on "drain" class
THRESH_RANGE       = np.linspace(0.0, 1.0, 101)

# Output CSVs
ENSEMBLE_SCAN_CSV  = "ensemble_threshold_scan.csv"
FINAL_METRICS_CSV  = "ensemble_pipeline_metrics.csv"

# 1. Load anomaly/drain annotations
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# 2. Collect device codes and split into train/validation sets
codes = [
    f.replace("device_","").replace(".csv","")
    for f in os.listdir(DATA_DIR)
    if f.startswith("device_") and f.endswith(".csv")
]
train_codes, val_codes = train_test_split(
    codes, train_size=TRAIN_DEVICE_COUNT, random_state=42
)

# 3. Define a function to load features and labels for a device
def load_data(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    # Basic features
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # Initialize label to normal
    df["label"] = 0
    # Mark anomaly segments
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        mask = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[mask, "label"] = 1
    # Mark drain segments
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        mask = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[mask, "label"] = 2
    # Drop rows without computed features
    df = df.dropna(subset=["diff","cum_drop"])
    return df[["diff","cum_drop"]], df["label"].astype(int)

# 4. Build training and validation datasets
def build_dataset(code_list):
    X_parts, y_parts = [], []
    for code in code_list:
        Xc, yc = load_data(code)
        X_parts.append(Xc)
        y_parts.append(yc)
    return pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True)

X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

# Prepare binary true labels for the "drain" class
y_true = (y_val == 2).astype(int)

# 5. Train & calibrate HistGradientBoostingClassifier (cost-sensitive)
hgb = HistGradientBoostingClassifier(
    loss="log_loss",
    class_weight=HGB_WEIGHT,
    max_iter=200,
    random_state=42
)
hgb.fit(X_train, y_train)
cal_hgb = CalibratedClassifierCV(hgb, cv=3, method="sigmoid")
cal_hgb.fit(X_train, y_train)
probs_hgb = cal_hgb.predict_proba(X_val)[:,2]

# 6. Train & calibrate online SGDClassifier (cost-sensitive)
sgd = SGDClassifier(
    loss="log_loss",
    class_weight=SGD_WEIGHT,
    max_iter=1,
    warm_start=True,
    random_state=42
)
for _ in range(10):
    sgd.partial_fit(X_train, y_train, classes=[0,1,2])
cal_sgd = CalibratedClassifierCV(sgd, cv="prefit", method="sigmoid")
cal_sgd.fit(X_train, y_train)
probs_sgd = cal_sgd.predict_proba(X_val)[:,2]

# 7. Ensemble threshold scan
scan_records = []
for t in THRESH_RANGE:
    ens_prob = HGB_WEIGHT_ENS * probs_hgb + SGD_WEIGHT_ENS * probs_sgd
    y_pred_bin = (ens_prob >= t).astype(int)
    p = precision_score(y_true, y_pred_bin, zero_division=0)
    r = recall_score(   y_true, y_pred_bin, zero_division=0)
    f = f1_score(       y_true, y_pred_bin, zero_division=0)
    scan_records.append({"threshold": t, "precision": p, "recall": r, "f1": f})

pd.DataFrame(scan_records).to_csv(ENSEMBLE_SCAN_CSV, index=False)

best = max(scan_records, key=lambda x: x["f1"])
th_ens = best["threshold"]

# 8. Final multi-class prediction
# Use HGB's multi-class prediction for non-drain, override to drain when ensemble prob exceeds threshold
base_pred = hgb.predict(X_val)
ens_prob  = HGB_WEIGHT_ENS * probs_hgb + SGD_WEIGHT_ENS * probs_sgd
y_pred_final = np.where(ens_prob >= th_ens, 2, base_pred)

# 9. Evaluation
print("=== Ensemble Pipeline ===")
print(f"Optimal threshold: {th_ens:.3f}", best)
print(classification_report(y_val, y_pred_final, target_names=["normal","anomaly","drain"]))

# 10. Save final metrics
metrics = {
    "threshold": th_ens,
    "precision": best["precision"],
    "recall":    best["recall"],
    "f1_score":  best["f1"],
    "support":   int(y_true.sum())
}
pd.DataFrame([metrics]).to_csv(FINAL_METRICS_CSV, index=False)

print(f"\n✅ Ensemble pipeline metrics saved to {FINAL_METRICS_CSV}")
