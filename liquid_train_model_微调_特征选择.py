#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# ===== Configuration =====
DATA_DIR           = "label_data"
EVENTS_FILE        = "labeled_events.csv"
TRAIN_DEVICE_COUNT = 30
WINDOW_LEN         = 5
DRAIN_WEIGHT       = 10
THRESHOLD_RANGE    = np.linspace(0.2, 0.8, 61)  # scan from 0.2 to 0.8
OUTPUT_REPORT_CSV  = "stacking_tuned_results.csv"

# 1. Load annotations
events = pd.read_csv(EVENTS_FILE, parse_dates=["start","end"])

# 2. List device codes and split
device_codes = [
    f.replace("device_","").replace(".csv","")
    for f in os.listdir(DATA_DIR)
    if f.startswith("device_") and f.endswith(".csv")
]
train_codes, val_codes = train_test_split(
    device_codes, train_size=TRAIN_DEVICE_COUNT, random_state=42
)

# 3. Feature extraction (rich features)
def load_features(code):
    path = os.path.join(DATA_DIR, f"device_{code}.csv")
    df = pd.read_csv(path, parse_dates=["msgTime"]).sort_values("msgTime").reset_index(drop=True)
    # basic
    df["diff"]     = df["liquidLevel_clean"].diff()
    df["cum_drop"] = df["diff"].rolling(window=WINDOW_LEN, min_periods=WINDOW_LEN).sum()
    # rolling stats
    for w in (3, 5, 12):
        df[f"max_drop_{w}h"]  = df["diff"].rolling(window=w, min_periods=1).min()
        df[f"var_slope_{w}h"] = df["diff"].rolling(window=w, min_periods=1).var()
    # time features
    df["hour"]       = df["msgTime"].dt.hour
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["weekday"]    = df["msgTime"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    # level bin
    df["level_bin"] = pd.qcut(df["liquidLevel_clean"], q=4, labels=False, duplicates="drop")
    # label
    df["label"] = 0
    for _, ev in events.query("code==@code and label=='anomaly'").iterrows():
        mask = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[mask, "label"] = 1
    for _, ev in events.query("code==@code and label=='drain'").iterrows():
        mask = (df["msgTime"] >= ev.start) & (df["msgTime"] <= ev.end)
        df.loc[mask, "label"] = 2
    df = df.dropna(subset=["diff","cum_drop","level_bin"])
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

# 4. Build train/val sets
X_train, y_train = build_dataset(train_codes)
X_val,   y_val   = build_dataset(val_codes)

# 5. Define Stacking with class_weight on final estimator
estimators = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        class_weight={0:1,1:1,2:DRAIN_WEIGHT},
        random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42))
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        class_weight={0:1,1:1,2:DRAIN_WEIGHT},
        max_iter=1000
    ),
    cv=5,
    n_jobs=-1
)

# 6. Hyperparameter tuning on stacking
param_grid = {
    "rf__n_estimators": [100, 200],
    "gb__learning_rate": [0.01, 0.1],
    "final_estimator__C": [0.1, 1, 10]
}
gs = GridSearchCV(stack, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
gs.fit(X_train, y_train)
best_stack = gs.best_estimator_
print("Best stacking params:", gs.best_params_)

# 7. Predict probabilities for drain
probs  = best_stack.predict_proba(X_val)[:,2]
y_true = (y_val == 2).astype(int)

# 8. Threshold tuning
records = []
for t in THRESHOLD_RANGE:
    y_pred_bin = (probs >= t).astype(int)
    p = precision_score(y_true, y_pred_bin, zero_division=0)
    r = recall_score(   y_true, y_pred_bin, zero_division=0)
    f = f1_score(       y_true, y_pred_bin, zero_division=0)
    records.append({"threshold": t, "precision": p, "recall": r, "f1": f})
df_thresh = pd.DataFrame(records)
best      = df_thresh.loc[df_thresh.f1.idxmax()]
print("Optimal threshold:", best.to_dict())

# 9. Final prediction & evaluation
y_pred_final = np.where(probs >= best.threshold,
                        2,
                        best_stack.predict(X_val))
print("\nFinal classification:")
print(classification_report(
    y_val, y_pred_final,
    target_names=["normal","anomaly","drain"]
))

# 10. Save results
df_thresh.to_csv("stacking_threshold_scan.csv", index=False)
pd.DataFrame(
    classification_report(
        y_val, y_pred_final,
        target_names=["normal","anomaly","drain"],
        output_dict=True
    )
).transpose().to_csv(OUTPUT_REPORT_CSV, encoding="utf-8-sig")
print(f"\n✅ Saved tuned stacking results to {OUTPUT_REPORT_CSV} and stacking_threshold_scan.csv")
