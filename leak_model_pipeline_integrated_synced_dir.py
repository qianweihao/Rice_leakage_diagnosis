#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
leak_model_pipeline_integrated_synced_dir.py
============================================
与 leak_model_pipeline_integrated_synced.py 相同逻辑，但支持 **直接读取文件夹里的每设备CSV**：
- 新增 --input_dir 与 --pattern（默认 *.csv）；也可继续用 --input 指定单一合并文件。
- 自动识别列名别名（device/code, timestamp/msgTime, level_mm/liquidLevel_clean），
  如CSV缺少设备列，会从文件名中提取（支持 device_XXXX.csv 或 XXXX.csv）。

两个入口（与原版一致）：
- events：从原始水位批量生成 labeled_events.csv（基于 liquidtest_sklearn 规则）
- train：读取设备逐时刻数据 + 事件文件，做分组评估并导出模型

用法示例：
1) 批量训练（文件夹输入）
python leak_model_pipeline_integrated_synced_dir.py train \
  --input_dir clean_results_smooth \
  --events path/to/labeled_events.csv \
  --freq 1H --use_raw_level

2) 单文件训练（保持兼容）
python leak_model_pipeline_integrated_synced_dir.py train \
  --input data/devices_all.parquet \
  --events path/to/labeled_events.csv \
  --freq 1H --use_raw_level
"""
import os, re, json, argparse
from glob import glob
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ---- 共用配置与工具（复用上一版逻辑，删减注释以便阅读） ----
RANDOM_STATE = 42
LEAK_CLASS = 2
FEATURE_COLUMNS_CANDIDATES = [
    'diff','drop1h','cum_drop_5','slope_3','rolling_min_5','rolling_max_5','rolling_std_5',
    'cum_drop_12','slope_5','rolling_std_12','hour','is_night'
]
DISCRETE_FEATURES = ['level_bin']

def ensure_datetime_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors='coerce', utc=True)

def resample_and_interpolate(df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = ensure_datetime_utc(df['timestamp'])
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df = df.resample(freq).mean().interpolate(limit=2)
    return df.reset_index()

def basic_feature_engineering(df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
    if 'level_mm' not in df.columns:
        raise ValueError("缺少 'level_mm' 列")
    df = resample_and_interpolate(df, freq=freq)
    base = df['level_mm']
    df['diff'] = base.diff()
    df['drop1h'] = (-df['diff']).clip(lower=0)
    df['cum_drop_5'] = df['drop1h'].rolling(5, min_periods=5).sum()
    df['slope_3'] = base.diff().rolling(3, min_periods=3).mean()
    df['rolling_min_5'] = base.rolling(5, min_periods=5).min()
    df['rolling_max_5'] = base.rolling(5, min_periods=5).max()
    df['rolling_std_5'] = base.rolling(5, min_periods=5).std()
    df['cum_drop_12'] = df['drop1h'].rolling(12, min_periods=12).sum()
    df['slope_5'] = base.diff().rolling(5, min_periods=5).mean()
    df['rolling_std_12'] = base.rolling(12, min_periods=12).std()
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = df['hour'].isin([22,23,0,1,2,3,4,5]).astype(int)
    try:
        df['level_bin'] = pd.qcut(base, q=10, labels=False, duplicates='drop')
    except Exception:
        pass
    df = df.dropna().reset_index(drop=True)
    return df

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in FEATURE_COLUMNS_CANDIDATES if c in df.columns]
    if 'level_bin' in df.columns and 'level_bin' not in cols:
        cols.append('level_bin')
    if not cols:
        raise ValueError("未检测到可用特征列")
    return cols

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    discrete_cols = [c for c in DISCRETE_FEATURES if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in discrete_cols]
    trs = []
    if numeric_cols: trs.append(("num", StandardScaler(), numeric_cols))
    if discrete_cols: trs.append(("cat", OneHotEncoder(handle_unknown='ignore'), discrete_cols))
    if not trs:
        return ColumnTransformer([('identity','passthrough', X.columns.tolist())])
    return ColumnTransformer(trs)

# ---- 事件生成（维持与 synced 版一致）简化：这里只保留接口占位，默认使用你已提供的 labeled_events.csv ----

# ---- 贴标签 ----
def tag_with_events(df_timeline: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    out = df_timeline.copy()
    out['timestamp'] = ensure_datetime_utc(out['timestamp'])
    out = out.sort_values(['device','timestamp']).reset_index(drop=True)
    out['label'] = 0
    ev = events_df.copy()
    if 'device' not in ev.columns and 'code' in ev.columns:
        ev = ev.rename(columns={'code':'device'})
    for col in ['start','end']:
        ev[col] = ensure_datetime_utc(ev[col])
    for lab_name, lab_id in [('anomaly',1), ('drain',2)]:
        sub = ev[ev['label'].str.lower() == lab_name]
        for dev, g in sub.groupby('device', sort=False):
            mdev = (out['device'] == str(dev))
            if not mdev.any(): continue
            ts = out.loc[mdev, 'timestamp']
            for _, row in g.iterrows():
                m = (ts >= row['start']) & (ts <= row['end'])
                idx = ts.index[m]
                out.loc[idx, 'label'] = lab_id
    return out

# ---- 评估 ----
@dataclass
class FoldResult:
    model_name: str; fold_idx: int; pr_auc_leak: float; best_thr: float; f1_at_best_thr: float

def evaluate_model_groupkfold(model_name, base_estimator, X, y, groups, n_splits=5, calibrate=True, calibration_method="isotonic"):
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]; ytr, yva = y[tr], y[va]
        clf = CalibratedClassifierCV(base_estimator, method=calibration_method, cv=3) if calibrate else base_estimator
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xva)
        p_leak = proba[:, 2]
        pr = average_precision_score((yva==2).astype(int), p_leak)
        thrs = np.linspace(0.05, 0.95, 19)
        best_f1, best_thr = -1.0, 0.5
        for t in thrs:
            yhat = np.argmax(proba, axis=1).copy()
            yhat[p_leak>=t] = 2
            f1 = f1_score((yva==2).astype(int), (yhat==2).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_thr = f1, t
        results.append(FoldResult(model_name, k, float(pr), float(best_thr), float(best_f1)))
    mean_pr = float(np.mean([r.pr_auc_leak for r in results]))
    mean_f1 = float(np.mean([r.f1_at_best_thr for r in results]))
    mean_thr = float(np.mean([r.best_thr for r in results]))
    return results, mean_pr, mean_f1, mean_thr

# ---- 读取 input_dir 并合并 ----
def infer_device_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r'device[_\-]?([A-Za-z0-9]+)', base)
    if m: return m.group(1)
    m2 = re.search(r'([A-Za-z0-9]+)\.csv$', base)
    if m2: return m2.group(1)
    return None

def normalize_columns(df, fallback_device=None):
    if 'device' not in df.columns:
        if 'code' in df.columns: df = df.rename(columns={'code':'device'})
        elif fallback_device is not None: df['device'] = fallback_device
        else: raise ValueError("缺少设备列(device/code)，且无法从文件名推断")
    if 'timestamp' not in df.columns:
        if 'msgTime' in df.columns: df = df.rename(columns={'msgTime':'timestamp'})
        else: raise ValueError("缺少时间列(timestamp/msgTime)")
    if 'level_mm' not in df.columns:
        if 'liquidLevel_clean' in df.columns: df = df.rename(columns={'liquidLevel_clean':'level_mm'})
        else: raise ValueError("缺少水位列(level_mm/liquidLevel_clean)")
    return df[['device','timestamp','level_mm']]

def read_input_any(input_path: str, input_dir: str = "", pattern: str = "*.csv") -> pd.DataFrame:
    if input_dir:
        files = sorted(glob(os.path.join(input_dir, pattern)))
        if not files: raise SystemExit(f"文件夹内未找到 {pattern}")
        parts = []
        for fp in files:
            dev = infer_device_from_filename(fp)
            df = pd.read_csv(fp)
            df = normalize_columns(df, fallback_device=dev)
            parts.append(df)
        all_df = pd.concat(parts, ignore_index=True)
    else:
        if input_path.lower().endswith(".parquet"):
            all_df = pd.read_parquet(input_path)
        else:
            all_df = pd.read_csv(input_path)
        all_df = normalize_columns(all_df)
    # 统一时区与排序
    all_df['timestamp'] = ensure_datetime_utc(all_df['timestamp'])
    all_df = all_df.dropna(subset=['timestamp']).sort_values(['device','timestamp']).reset_index(drop=True)
    return all_df

# ---- 训练 ----
def train_main(args):
    outdir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    raw = read_input_any(args.input, args.input_dir, args.pattern)

    # 从文件或文件夹读取事件
    if not args.events:
        raise ValueError("该目录友好版不支持自动造事件，请提供 --events")
    ev = pd.read_parquet(args.events) if args.events.lower().endswith(".parquet") else pd.read_csv(args.events)

    # 等频+特征（逐设备）
    parts = []
    for dev, g in raw.groupby('device', sort=False):
        g2 = g[['timestamp','level_mm']].copy()
        feats = basic_feature_engineering(g2, freq=args.freq) if args.use_raw_level else resample_and_interpolate(g2, freq=args.freq)
        feats['device'] = str(dev)
        parts.append(feats)
    timeline = pd.concat(parts, ignore_index=True)

    # 贴标签
    labeled = tag_with_events(timeline, ev)

    # 训练/评估
    feature_cols = infer_feature_columns(labeled)
    X_all = labeled[feature_cols].copy()
    y_all = labeled['label'].astype(int).values
    groups = labeled['device'].astype(str).values

    et = ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                              random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1)
    preproc = build_preprocessor(X_all)
    lr = Pipeline([('prep', preproc), ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                                                max_iter=200, class_weight='balanced',
                                                                random_state=RANDOM_STATE))])
    candidates = {"ExtraTrees": et, "RandomForest": rf, "LogReg": lr}

    model_cv_results: Dict[str, Dict] = {}
    rows = []
    for name, est in candidates.items():
        folds, mean_pr, mean_f1, mean_thr = evaluate_model_groupkfold(name, est, X_all, y_all, groups,
                                                                      n_splits=args.n_splits, calibrate=True)
        model_cv_results[name] = {"folds":[r.__dict__ for r in folds], "mean_pr_auc":mean_pr,
                                  "mean_f1":mean_f1, "mean_thr":mean_thr}
        rows.append([name, mean_pr, mean_f1, mean_thr])

    summary_df = pd.DataFrame(rows, columns=['model','mean_pr_auc_leak','mean_f1_at_best_thr','mean_best_thr'])
    summary_df.to_csv(os.path.join(outdir, "cv_summary.csv"), index=False)
    with open(os.path.join(outdir, "cv_details.json"), 'w', encoding='utf-8') as f:
        json.dump(model_cv_results, f, ensure_ascii=False, indent=2)

    def rank_key(kv): name, s = kv; return (s["mean_pr_auc"], s["mean_f1"], 1 if name=="LogReg" else 0)
    best_name, best_stats = sorted(model_cv_results.items(), key=rank_key, reverse=True)[0]
    best_est = candidates[best_name]
    best_clf = CalibratedClassifierCV(best_est, method='isotonic', cv=3).fit(X_all, y_all)

    joblib.dump({
        "pipeline": best_clf,
        "feature_cols": feature_cols,
        "use_raw_level": bool(args.use_raw_level),
        "freq": args.freq,
        "meta": {"model_name": best_name, "trained_at": datetime.now().isoformat()}
    }, os.path.join(outdir, "best_pipeline.pkl"))

    with open(os.path.join(outdir, "best_thresholds.json"), 'w', encoding='utf-8') as f:
        json.dump({"type":"single","best_model":best_name,"leak_threshold":float(best_stats["mean_thr"])}, f, ensure_ascii=False, indent=2)

    print("=== CV Summary ===")
    print(summary_df.sort_values('mean_pr_auc_leak', ascending=False).to_string(index=False))
    print(f"\nChampion: {best_name} | PR-AUC={best_stats['mean_pr_auc']:.4f} | F1*={best_stats['mean_f1']:.4f} | Thr~{best_stats['mean_thr']:.2f}")
    print(f"Artifacts saved under: {outdir}")

# ---- 推理（保持单文件/合并文件输入；如需文件夹推理，可先用 combine_devices.py 合并） ----
def postprocess_events(timeline: pd.DataFrame, min_len=3, gap_merge=1) -> pd.DataFrame:
    tl = timeline.copy().sort_values('timestamp')
    tl['pred_change'] = (tl['pred'] != tl['pred'].shift()).astype(int)
    tl['block'] = tl['pred_change'].cumsum()
    events = []
    for _, g in tl.groupby('block'):
        label = int(g['pred'].iloc[0])
        if label != 2 or len(g) < min_len: continue
        events.append({"start": g['timestamp'].iloc[0], "end": g['timestamp'].iloc[-1], "n_steps": len(g)})
    if not events: return pd.DataFrame(columns=['start','end','n_steps'])
    ev = pd.DataFrame(events).sort_values('start').reset_index(drop=True)
    merged = []; cur = ev.iloc[0].to_dict()
    for i in range(1, len(ev)):
        row = ev.iloc[i]; gap = (row['start'] - cur['end']) / np.timedelta64(1, 'h')
        if gap <= gap_merge:
            cur['end'] = max(cur['end'], row['end']); cur['n_steps'] += row['n_steps']
        else:
            merged.append(cur); cur = row.to_dict()
    merged.append(cur); return pd.DataFrame(merged)

def infer_main(args):
    # 这里保持与原版一致：接收单一 --input 文件；若你有多CSV，先合并或后续可扩展
    if args.input.lower().endswith(".parquet"): df = pd.read_parquet(args.input)
    else: df = pd.read_csv(args.input)
    if 'device' not in df.columns and 'code' in df.columns: df = df.rename(columns={'code':'device'})
    if 'timestamp' not in df.columns and 'msgTime' in df.columns: df = df.rename(columns={'msgTime':'timestamp'})
    if 'level_mm' not in df.columns and 'liquidLevel_clean' in df.columns: df = df.rename(columns={'liquidLevel_clean':'level_mm'})
    model_obj = joblib.load(args.model)

    parts = []
    if model_obj.get("use_raw_level", False) or args.use_raw_level:
        for dev, g in df.groupby('device', sort=False):
            feats = basic_feature_engineering(g[['timestamp','level_mm']], freq=model_obj.get('freq', args.freq))
            feats['device'] = str(dev); parts.append(feats)
    else:
        parts.append(df.copy())
    feat = pd.concat(parts, ignore_index=True)

    feature_cols = model_obj.get('feature_cols', infer_feature_columns(feat))
    X = feat[feature_cols].copy()
    if 'pipelines' in model_obj: raise NotImplementedError("本简版未包含集成推理；请使用best_pipeline.pkl")
    proba = model_obj['pipeline'].predict_proba(X)
    thr = float(json.load(open(args.thr_json,'r',encoding='utf-8')).get('leak_threshold', args.threshold)) if args.thr_json else float(args.threshold)

    pred = np.argmax(proba, axis=1); leak_mask = proba[:,2] >= thr; pred[leak_mask] = 2
    timeline = feat[['timestamp']].copy(); timeline['pred'] = pred; timeline['p_leak'] = proba[:,2]
    if 'device' in df.columns: timeline['device'] = df['device'].astype(str).values[:len(timeline)]
    events = postprocess_events(timeline, min_len=args.min_steps, gap_merge=args.gap_merge)
    if args.timeline: timeline.to_csv(args.timeline, index=False)
    if args.save: events.to_csv(args.save, index=False)
    print("Inference done.")
    print("\n[Timeline head]\n", timeline.head().to_string(index=False))
    print("\n[Events]\n", events.to_string(index=False))

# ---- CLI ----
def build_parser():
    p = argparse.ArgumentParser(description="Leak Detection - Integrated (folder-friendly)")
    sub = p.add_subparsers(dest='cmd')

    # 训练（支持 --input 或 --input_dir）
    p_tr = sub.add_parser('train', help='Train with GroupKFold using events')
    p_tr.add_argument('--input', default='', help='单一CSV/Parquet（可留空改用 --input_dir）')
    p_tr.add_argument('--input_dir', default='', help='包含每设备CSV的目录，如 clean_results_smooth')
    p_tr.add_argument('--pattern', default='*.csv', help='匹配模式，默认 *.csv')
    p_tr.add_argument('--events', required=True, help='labeled_events.csv / parquet')
    p_tr.add_argument('--n_splits', type=int, default=5, help='GroupKFold 折数')
    p_tr.add_argument('--freq', type=str, default='1H', help='等频重采样频率')
    p_tr.add_argument('--use_raw_level', action='store_true', help='基于 level_mm 自动造特征（推荐）')

    # 推理（仍用单文件；多CSV请先合并）
    p_in = sub.add_parser('infer', help='Run inference on new device data')
    p_in.add_argument('--input', required=True, help='CSV/Parquet（至少 timestamp,device,level_mm 或别名）')
    p_in.add_argument('--model', required=True, help='best_pipeline.pkl')
    p_in.add_argument('--thr_json', default='', help='best_thresholds.json（若缺省使用 --threshold）')
    p_in.add_argument('--threshold', type=float, default=0.5, help='漏水概率阈值')
    p_in.add_argument('--freq', type=str, default='1H', help='特征构造频率')
    p_in.add_argument('--use_raw_level', action='store_true', help='与训练口径一致')
    p_in.add_argument('--min_steps', type=int, default=3, help='事件最小持续步数')
    p_in.add_argument('--gap_merge', type=int, default=1, help='相邻事件合并最大间隔步数')
    p_in.add_argument('--save', default='', help='保存事件CSV路径')
    p_in.add_argument('--timeline', default='', help='保存逐时刻预测CSV路径')
    return p

def main():
    parser = build_parser(); args = parser.parse_args()
    if args.cmd == 'train': train_main(args)
    elif args.cmd == 'infer': infer_main(args)
    else: parser.print_help()

if __name__ == '__main__':
    main()
