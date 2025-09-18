#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
leak_model_pipeline_integrated_synced.py
===================================================
一体化脚本（含 liquidtest_sklearn 事件提取规则）：
- 事件生成(events)：根据“夜间急升=异常；连续下降=排水；三道过滤；异常后5小时屏蔽排水”的规则，
  从原始水位数据生成 labeled_events.csv（字段：code,start,end,label）。
- 训练(train)：可直接读原始水位 + --events（或 --gen_events 自动生成）→ 等频 → 造特征 → 贴标签 →
  按设备分组评估 → 导出冠军模型/阈值/可选软投票集成。
- 推理(infer)：加载导出的模型(pkl)与阈值(json)，对新设备时序做预测并输出事件表。

典型用法：
1) 直接一条龙：原始水位 -> 自动生成事件 -> 训练
python leak_model_pipeline_integrated_synced.py train \
  --input data/waterlevel.csv \
  --gen_events \
  --freq 1H --use_raw_level

2) 若你已有 labeled_events.csv
python leak_model_pipeline_integrated_synced.py train \
  --input data/waterlevel.csv \
  --events data/labeled_events.csv \
  --freq 1H --use_raw_level

3) 单独生成事件文件
python leak_model_pipeline_integrated_synced.py events \
  --input data/waterlevel.csv \
  --output data/labeled_events.csv

4) 推理
python leak_model_pipeline_integrated_synced.py infer \
  --input data/new_device.csv \
  --model outputs/xxxx/best_pipeline.pkl \
  --thr_json outputs/xxxx/best_thresholds.json \
  --freq 1H --use_raw_level \
  --save events.csv --timeline timeline.csv
"""
import os
import json
import argparse
from datetime import datetime, timedelta
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

RANDOM_STATE = 42
LEAK_CLASS = 2
CLASS_NAMES = {0: 'normal', 1: 'anomaly', 2: 'drain'}
LABEL_MAP = {"normal": 0, "anomaly": 1, "drain": 2}

# 候选特征（若不存在会基于 level_mm 自动构造最小特征）
FEATURE_COLUMNS_CANDIDATES = [
    'diff', 'drop1h', 'cum_drop_5', 'slope_3', 'rolling_min_5', 'rolling_max_5', 'rolling_std_5',
    'cum_drop_12', 'slope_5', 'rolling_std_12',
    'hour', 'is_night'
]
DISCRETE_FEATURES = ['level_bin']  # 若存在将做One-Hot

# ===================== 基础工具 =====================
def ensure_datetime_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors='coerce', utc=True)

def resample_and_interpolate(df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = ensure_datetime_utc(df['timestamp'])
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').set_index('timestamp')
    df = df.resample(freq).mean().interpolate(limit=2)
    return df.reset_index()

def basic_feature_engineering(df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
    """基于 level_mm 造最小可用特征；若你有更完整特征，请在此补充保持一致"""
    if 'level_mm' not in df.columns:
        raise ValueError("缺少 'level_mm' 列，无法构造特征；请去掉 --use_raw_level 或提供该列。")
    df = resample_and_interpolate(df, freq=freq)
    # 差分与下降
    df['diff'] = df['level_mm'].diff()
    df['drop1h'] = (-df['diff']).clip(lower=0)
    # 滚动（等频前提下的样本窗）
    df['cum_drop_5']  = df['drop1h'].rolling(5,  min_periods=5).sum()
    df['slope_3']     = df['level_mm'].diff().rolling(3, min_periods=3).mean()
    df['rolling_min_5'] = df['level_mm'].rolling(5,  min_periods=5).min()
    df['rolling_max_5'] = df['level_mm'].rolling(5,  min_periods=5).max()
    df['rolling_std_5'] = df['level_mm'].rolling(5,  min_periods=5).std()
    df['cum_drop_12'] = df['drop1h'].rolling(12, min_periods=12).sum()
    df['slope_5']     = df['level_mm'].diff().rolling(5, min_periods=5).mean()
    df['rolling_std_12'] = df['level_mm'].rolling(12, min_periods=12).std()
    # 时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = df['hour'].isin([22,23,0,1,2,3,4,5]).astype(int)
    # 分桶（可选）
    try:
        df['level_bin'] = pd.qcut(df['level_mm'], q=10, labels=False, duplicates='drop')
    except Exception:
        pass
    df = df.dropna().reset_index(drop=True)
    return df

def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in FEATURE_COLUMNS_CANDIDATES if c in df.columns]
    if not cols:
        raise ValueError("未检测到可用特征列，请提供特征或使用 --use_raw_level 让脚本构造。")
    if 'level_bin' in df.columns and 'level_bin' not in cols:
        cols.append('level_bin')
    return cols

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    discrete_cols = [c for c in DISCRETE_FEATURES if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in discrete_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if discrete_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore'), discrete_cols))
    if not transformers:
        return ColumnTransformer([('identity', 'passthrough', X.columns.tolist())])
    return ColumnTransformer(transformers)

# ===================== liquidtest_sklearn 事件规则 =====================
# 默认阈值（可通过 CLI 覆盖）
RISE_TH = 4.0          # 夜间急升阈值 mm/h
DROP_TH = 1.2          # 每小时下降阈值 mm/h（diff < -DROP_TH 记为明显下降）
DROP_SUM_TH = 8.0      # DRAIN_LEN_H 小时内累计下降阈值 mm
DRAIN_LEN_H = 5        # 连续小时数
SPIKE_TH = 30.0        # “起点 vs 前5小时”突变过滤阈值 mm
POST_ANOMALY_BLOCK_H = 5  # 异常结束后屏蔽排水的小时数

def extract_segments_from_mask(ts: pd.Series, mask: np.ndarray, min_len: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """根据布尔序列提取连续True片段，返回[(start,end)]（包含端点）。"""
    segments = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            end = i - 1
            if end - start + 1 >= min_len:
                segments.append((ts.iloc[start], ts.iloc[end]))
            start = None
    if start is not None:
        end = len(mask) - 1
        if end - start + 1 >= min_len:
            segments.append((ts.iloc[start], ts.iloc[end]))
    return segments

def build_events_from_raw(df_raw: pd.DataFrame,
                          code_col: str = 'code',
                          time_col: str = 'msgTime',
                          level_col: str = 'liquidLevel_clean',
                          freq: str = '1H',
                          rise_th: float = RISE_TH,
                          drop_th: float = DROP_TH,
                          drop_sum_th: float = DROP_SUM_TH,
                          drain_len_h: int = DRAIN_LEN_H,
                          spike_th: float = SPIKE_TH,
                          post_anom_block_h: int = POST_ANOMALY_BLOCK_H) -> pd.DataFrame:
    """
    将原始水位（可能不等频）转换为事件表 labeled_events：
    规则：
      - anomaly：夜间(22:00–05:00) 且 diff > rise_th
      - drain：连续 drain_len_h 小时 diff < -drop_th ；或 drain_len_h 内累计下降 < -drop_sum_th
      - 过滤1：abs(level_start - level_prev5) > spike_th → 忽略
      - 过滤2：(level_prev1 > level_start) 且 (level_prev1 > level_prev5) → 忽略
      - 过滤3：若 drain.start 落在任一 anomaly.end 后 post_anom_block_h 小时内 → 忽略
    返回：DataFrame[code,start,end,label]
    """
    need_cols = {code_col, time_col, level_col}
    if not need_cols.issubset(set(df_raw.columns)):
        raise ValueError(f"输入缺少列：{need_cols - set(df_raw.columns)}")

    df_raw = df_raw.copy()
    df_raw[time_col] = ensure_datetime_utc(df_raw[time_col])
    records = []
    # 先收集所有 anomaly 结束时间，供“过滤3”使用
    anomaly_map = {}

    for code, g in df_raw.groupby(code_col, sort=False):
        g = g[[time_col, level_col]].dropna().sort_values(time_col).reset_index(drop=True)
        # 等频
        g = g.set_index(time_col).resample(freq).mean().interpolate(limit=2).reset_index()
        g.rename(columns={time_col: 'timestamp', level_col: 'level'}, inplace=True)
        # 计算 diff & 夜间
        g['diff'] = g['level'].diff()
        hrs = g['timestamp'].dt.hour
        g['is_night'] = (hrs >= 22) | (hrs <= 5)
        # anomaly 片段
        anomaly_mask = g['is_night'] & (g['diff'] > rise_th)
        anomaly_segs = extract_segments_from_mask(g['timestamp'], anomaly_mask.values, min_len=1)
        anomaly_map[code] = [e for (_, e) in anomaly_segs]
        for (s, e) in anomaly_segs:
            records.append({"code": code, "start": s, "end": e, "label": "anomaly"})

    # 第二遍生成 drain（需要已知 anomaly 结束以做过滤3）
    for code, g in df_raw.groupby(code_col, sort=False):
        g = g[[time_col, level_col]].dropna().sort_values(time_col).reset_index(drop=True)
        g = g.set_index(time_col).resample(freq).mean().interpolate(limit=2).reset_index()
        g.rename(columns={time_col: 'timestamp', level_col: 'level'}, inplace=True)
        g['diff'] = g['level'].diff().fillna(0.0)

        # 以“每小时明显下降”为基础掩码
        down_mask = g['diff'] < (-drop_th)
        # 连续 drain_len_h 小时明显下降 → 候选
        down_runs = extract_segments_from_mask(g['timestamp'], down_mask.values, min_len=drain_len_h)

        # 累计下降判断（滑窗求和）
        roll_drop = (-g['diff']).clip(lower=0).rolling(drain_len_h, min_periods=drain_len_h).sum().fillna(0.0)
        sum_mask = roll_drop > drop_sum_th
        sum_runs = extract_segments_from_mask(g['timestamp'], sum_mask.values, min_len=1)

        # 合并两种候选区间（按时间并集粗略合并）
        candidates = sorted(down_runs + sum_runs, key=lambda x: x[0])

        # 过滤与去重
        seen = []
        for (start, end) in candidates:
            # 去重：与已收集片段重叠则跳过
            if any((s <= end and e >= start) for (s, e) in seen):
                continue

            # 过滤1：起点 vs 前5小时突变
            idx_start = g.index[g['timestamp'] == start][0]
            idx_prev5 = max(0, idx_start - 5)
            level_start = g.at[idx_start, 'level']
            level_prev5 = g.at[idx_prev5, 'level']
            if abs(level_start - level_prev5) > spike_th:
                continue

            # 过滤2：前1小时 vs 前5小时的“高点”
            idx_prev1 = max(0, idx_start - 1)
            level_prev1 = g.at[idx_prev1, 'level']
            if (level_prev1 > level_start) and (level_prev1 > level_prev5):
                continue

            # 过滤3：距离最近 anomaly.end 是否在 post_anom_block_h 小时内
            ok = True
            for ae in anomaly_map.get(code, []):
                if start <= ae + pd.Timedelta(hours=post_anom_block_h):
                    # 只有当 start 在该异常结束后的屏蔽窗口内，并且 start > ae（位于异常之后）才拦截
                    if start > ae:
                        ok = False
                        break
            if not ok:
                continue

            seen.append((start, end))
            records.append({"code": code, "start": start, "end": end, "label": "drain"})

    if records:
        out_df = pd.DataFrame(records).sort_values(['code','start','end']).reset_index(drop=True)
    else:
        out_df = pd.DataFrame(columns=['code','start','end','label'])
    return out_df

# ===================== 事件贴标 =====================
def tag_with_events(df_timeline: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    在等频时间线上贴 labels（0/1/2）；冲突时“drain > anomaly”。
    df_timeline: 列至少 ['timestamp','device']；需已等频（或近似等频）。
    events_df:   列 ['code/device','start','end','label']，label ∈ {'anomaly','drain'}。
    """
    out = df_timeline.copy()
    out['timestamp'] = ensure_datetime_utc(out['timestamp'])
    out = out.sort_values(['device','timestamp']).reset_index(drop=True)
    out['label'] = 0

    ev = events_df.copy()
    if 'device' not in ev.columns and 'code' in ev.columns:
        ev = ev.rename(columns={'code': 'device'})
    if 'device' not in ev.columns:
        raise ValueError("events文件需要包含 'device' 或 'code' 列")
    for col in ['start','end']:
        ev[col] = ensure_datetime_utc(ev[col])

    # anomaly 后 drain 覆盖
    for lab_name, lab_id in [('anomaly', 1), ('drain', 2)]:
        sub = ev[ev['label'].str.lower() == lab_name]
        if sub.empty:
            continue
        for dev, g in sub.groupby('device', sort=False):
            mask_dev = (out['device'] == str(dev))
            if not mask_dev.any():
                continue
            ts = out.loc[mask_dev, 'timestamp']
            for _, row in g.iterrows():
                m = (ts >= row['start']) & (ts <= row['end'])
                idx = ts.index[m]
                out.loc[idx, 'label'] = lab_id
    return out

# ===================== 评估（GroupKFold） =====================
@dataclass
class FoldResult:
    model_name: str
    fold_idx: int
    pr_auc_leak: float
    best_thr: float
    f1_at_best_thr: float

def evaluate_model_groupkfold(
    model_name: str,
    base_estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> Tuple[List[FoldResult], float, float, float]:
    gkf = GroupKFold(n_splits=n_splits)
    fold_results: List[FoldResult] = []
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if calibrate:
            clf = CalibratedClassifierCV(base_estimator, method=calibration_method, cv=3)
        else:
            clf = base_estimator
        clf.fit(X_tr, y_tr)
        proba_val = clf.predict_proba(X_val)
        p_leak = proba_val[:, LEAK_CLASS]
        pr_auc = average_precision_score((y_val == LEAK_CLASS).astype(int), p_leak)
        thrs = np.linspace(0.05, 0.95, 19)
        best_f1, best_thr = -1.0, 0.5
        for t in thrs:
            y_pred = np.argmax(proba_val, axis=1).copy()
            leak_mask = p_leak >= t
            y_pred[leak_mask] = LEAK_CLASS
            f1 = f1_score((y_val == LEAK_CLASS).astype(int), (y_pred == LEAK_CLASS).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        fold_results.append(FoldResult(model_name, fold_idx, pr_auc, best_thr, best_f1))
    mean_pr_auc = float(np.mean([r.pr_auc_leak for r in fold_results]))
    mean_f1 = float(np.mean([r.f1_at_best_thr for r in fold_results]))
    mean_thr = float(np.mean([r.best_thr for r in fold_results]))
    return fold_results, mean_pr_auc, mean_f1, mean_thr

# ===================== 训练主流程 =====================
def train_main(args):
    outdir = os.path.join("outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    # 读取输入
    if args.input.lower().endswith(".parquet"):
        raw = pd.read_parquet(args.input)
    else:
        raw = pd.read_csv(args.input)

    # 统一列名（兼容 code/msgTime/liquidLevel_clean）
    # device列：优先 device，否则 code
    if 'device' not in raw.columns and 'code' in raw.columns:
        raw = raw.rename(columns={'code':'device'})
    if 'device' not in raw.columns:
        raise ValueError("输入数据需要包含 'device' 或 'code' 列")
    # timestamp列：优先 timestamp，否则 msgTime
    if 'timestamp' not in raw.columns and 'msgTime' in raw.columns:
        raw = raw.rename(columns={'msgTime':'timestamp'})
    if 'timestamp' not in raw.columns:
        raise ValueError("输入数据需要包含 'timestamp' 或 'msgTime' 列")
    # level列：优先 level_mm，否则 liquidLevel_clean
    if 'level_mm' not in raw.columns and 'liquidLevel_clean' in raw.columns:
        raw = raw.rename(columns={'liquidLevel_clean':'level_mm'})

    # 若要求自动生成事件
    if args.gen_events and not args.events:
        events_df = build_events_from_raw(
            df_raw=raw.rename(columns={'device':'code', 'timestamp':'msgTime', 'level_mm':'liquidLevel_clean'}),
            code_col='code', time_col='msgTime', level_col='liquidLevel_clean',
            freq=args.freq, rise_th=args.rise_th, drop_th=args.drop_th,
            drop_sum_th=args.drop_sum_th, drain_len_h=args.drain_len_h,
            spike_th=args.spike_th, post_anom_block_h=args.post_anom_block_h
        )
        events_path = os.path.join(outdir, "labeled_events.csv")
        events_df.to_csv(events_path, index=False, encoding='utf-8-sig')
        args.events = events_path
        print(f"[info] 已基于 liquidtest_sklearn 规则生成事件：{events_path}（共{len(events_df)}条）")

    # 若提供事件文件，则：等频→特征→贴标签
    if args.events:
        # 逐设备：等频→特征→合并
        parts = []
        for dev, g in raw.groupby('device', sort=False):
            g = g[['timestamp','device'] + (['level_mm'] if 'level_mm' in g.columns else [])].copy()
            if args.use_raw_level:
                feats = basic_feature_engineering(g[['timestamp','level_mm']], freq=args.freq)
            else:
                feats = resample_and_interpolate(g[['timestamp','level_mm']], freq=args.freq) if 'level_mm' in g.columns else g.copy()
            feats['device'] = str(dev)
            parts.append(feats)
        timeline_all = pd.concat(parts, ignore_index=True)

        # 读事件文件
        if args.events.lower().endswith(".parquet"):
            ev = pd.read_parquet(args.events)
        else:
            ev = pd.read_csv(args.events)
        labeled = tag_with_events(timeline_all, ev)
    else:
        # 未给事件文件：要求已有逐时刻 label
        if 'label' not in raw.columns:
            raise ValueError("未提供 --events/--gen_events 时，输入数据必须包含逐时刻 'label' 列。")
        labeled = raw.copy()
        if args.use_raw_level and 'level_mm' not in labeled.columns:
            raise ValueError("使用 --use_raw_level 训练时需要 'level_mm' 列。")

    # 如需造特征（若 events 分支里已造则跳过）
    if args.use_raw_level and ('diff' not in labeled.columns):
        parts2 = []
        for dev, g in labeled.groupby('device', sort=False):
            feats = basic_feature_engineering(g[['timestamp','level_mm']], freq=args.freq)
            feats['device'] = str(dev)
            merged = pd.merge_asof(
                feats.sort_values('timestamp'),
                g[['timestamp','label']].sort_values('timestamp'),
                on='timestamp', direction='nearest', tolerance=pd.Timedelta(args.freq)
            ).dropna(subset=['label'])
            parts2.append(merged)
        labeled = pd.concat(parts2, ignore_index=True)

    feature_cols = infer_feature_columns(labeled)
    X_all = labeled[feature_cols].copy()
    y_all = labeled['label'].astype(int).values
    groups = labeled['device'].astype(str).values

    # 候选模型
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1
    )
    preproc = build_preprocessor(X_all)
    lr = Pipeline(steps=[
        ('prep', preproc),
        ('clf', LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=200,
            class_weight='balanced', random_state=RANDOM_STATE
        ))
    ])
    candidates = {"ExtraTrees": et, "RandomForest": rf, "LogReg": lr}

    # 逐模型评估
    model_cv_results: Dict[str, Dict] = {}
    rows = []
    for name, est in candidates.items():
        folds, mean_pr, mean_f1, mean_thr = evaluate_model_groupkfold(
            model_name=name, base_estimator=est,
            X=X_all, y=y_all, groups=groups,
            n_splits=args.n_splits, calibrate=True, calibration_method='isotonic'
        )
        model_cv_results[name] = {
            "folds": [r.__dict__ for r in folds],
            "mean_pr_auc": mean_pr,
            "mean_f1": mean_f1,
            "mean_thr": mean_thr
        }
        rows.append([name, mean_pr, mean_f1, mean_thr])

    summary_df = pd.DataFrame(rows, columns=['model','mean_pr_auc_leak','mean_f1_at_best_thr','mean_best_thr'])
    summary_csv = os.path.join(outdir, "cv_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    with open(os.path.join(outdir, "cv_details.json"), 'w', encoding='utf-8') as f:
        json.dump(model_cv_results, f, ensure_ascii=False, indent=2)

    # 选冠军（PR-AUC优先，近似平手看F1，再偏好简单模型）
    def model_rank_key(item):
        name, stats = item
        return (stats["mean_pr_auc"], stats["mean_f1"], 1 if name=="LogReg" else 0)
    best_name, best_stats = sorted(model_cv_results.items(), key=model_rank_key, reverse=True)[0]

    print("=== CV Summary ===")
    print(summary_df.sort_values('mean_pr_auc_leak', ascending=False).to_string(index=False))
    print(f"\nChampion: {best_name} | PR-AUC={best_stats['mean_pr_auc']:.4f} | F1*={best_stats['mean_f1']:.4f} | Thr~{best_stats['mean_thr']:.2f}")

    # 训练全量冠军 + 校准 → 导出
    best_est = candidates[best_name]
    best_clf = CalibratedClassifierCV(best_est, method='isotonic', cv=3).fit(X_all, y_all)
    best_pipeline_path = os.path.join(outdir, "best_pipeline.pkl")
    joblib.dump({
        "pipeline": best_clf,
        "feature_cols": feature_cols,
        "use_raw_level": bool(args.use_raw_level),
        "freq": args.freq,
        "meta": {"model_name": best_name, "trained_at": datetime.now().isoformat()}
    }, best_pipeline_path)

    # 若次优接近，导出软投票
    sorted_models = sorted(model_cv_results.items(), key=lambda kv: kv[1]["mean_pr_auc"], reverse=True)
    ensemble_used = False
    if len(sorted_models) >= 2:
        (name1, stats1), (name2, stats2) = sorted_models[0], sorted_models[1]
        if (stats1["mean_pr_auc"] - stats2["mean_pr_auc"]) < 0.02:
            clf1 = CalibratedClassifierCV(candidates[name1], method='isotonic', cv=3).fit(X_all, y_all)
            clf2 = CalibratedClassifierCV(candidates[name2], method='isotonic', cv=3).fit(X_all, y_all)
            joblib.dump({
                "pipelines": {name1: clf1, name2: clf2},
                "weights": {name1: float(stats1["mean_pr_auc"]), name2: float(stats2["mean_pr_auc"])},
                "feature_cols": feature_cols,
                "use_raw_level": bool(args.use_raw_level),
                "freq": args.freq,
                "meta": {"type":"soft_voting","trained_at": datetime.now().isoformat(),"members":[name1,name2]}
            }, os.path.join(outdir, "ensemble_soft_voting.pkl"))
            ensemble_used = True

    with open(os.path.join(outdir, "best_thresholds.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "type": "single" if not ensemble_used else "maybe_ensemble",
            "best_model": best_name,
            "leak_threshold": float(best_stats["mean_thr"])
        }, f, ensure_ascii=False, indent=2)

    print(f"\nArtifacts saved to: {outdir}")
    print(f"- CV summary: {summary_csv}")
    print(f"- Champion pipeline: {best_pipeline_path}")
    if ensemble_used:
        print(f"- Soft-voting ensemble exported.")
    print(f"- Thresholds: {os.path.join(outdir, 'best_thresholds.json')}")

# ===================== 推理 & 事件合并 =====================
def postprocess_events(timeline: pd.DataFrame, min_len: int = 3, gap_merge: int = 1) -> pd.DataFrame:
    tl = timeline.copy().sort_values('timestamp')
    tl['pred_change'] = (tl['pred'] != tl['pred'].shift()).astype(int)
    tl['block'] = tl['pred_change'].cumsum()

    events = []
    for _, g in tl.groupby('block'):
        label = int(g['pred'].iloc[0])
        if label != LEAK_CLASS:
            continue
        if len(g) < min_len:
            continue
        events.append({"start": g['timestamp'].iloc[0], "end": g['timestamp'].iloc[-1], "n_steps": len(g)})
    if not events:
        return pd.DataFrame(columns=['start','end','n_steps']).astype({'start':'datetime64[ns, UTC]','end':'datetime64[ns, UTC]','n_steps':'int64'})

    ev = pd.DataFrame(events).sort_values('start').reset_index(drop=True)
    # 合并间隔小的事件
    merged = []
    cur = ev.iloc[0].to_dict()
    for i in range(1, len(ev)):
        row = ev.iloc[i]
        gap = (row['start'] - cur['end']) / np.timedelta64(1, 'h')
        if gap <= gap_merge:
            cur['end'] = max(cur['end'], row['end'])
            cur['n_steps'] += row['n_steps']
        else:
            merged.append(cur)
            cur = row.to_dict()
    merged.append(cur)
    return pd.DataFrame(merged)

def soft_voting_predict_proba(ensemble_obj: dict, X: pd.DataFrame) -> np.ndarray:
    pipelines = ensemble_obj['pipelines']
    weights = ensemble_obj['weights']
    weight_sum = sum(weights.values())
    proba_sum = None
    for name, pipe in pipelines.items():
        p = pipe.predict_proba(X)
        w = weights[name] / weight_sum
        proba_sum = p * w if proba_sum is None else (proba_sum + p * w)
    return proba_sum

def infer_main(args):
    # 读新数据
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # 兼容列名
    if 'device' not in df.columns and 'code' in df.columns:
        df = df.rename(columns={'code':'device'})
    if 'timestamp' not in df.columns and 'msgTime' in df.columns:
        df = df.rename(columns={'msgTime':'timestamp'})
    if 'level_mm' not in df.columns and 'liquidLevel_clean' in df.columns:
        df = df.rename(columns={'liquidLevel_clean':'level_mm'})

    model_obj = joblib.load(args.model)

    # 特征准备
    if model_obj.get("use_raw_level", False) or args.use_raw_level:
        if 'level_mm' not in df.columns:
            raise ValueError("推理需要 'level_mm'（导出模型声明使用raw level造特征）。")
        parts = []
        for dev, g in df.groupby('device', sort=False):
            feats = basic_feature_engineering(g[['timestamp','level_mm']], freq=model_obj.get('freq', args.freq))
            feats['device'] = str(dev)
            parts.append(feats)
        feat = pd.concat(parts, ignore_index=True)
    else:
        feat = df.copy()

    feature_cols = model_obj.get('feature_cols', infer_feature_columns(feat))
    X = feat[feature_cols].copy()

    # 概率
    if 'pipelines' in model_obj:  # ensemble
        proba = soft_voting_predict_proba(model_obj, X)
        meta = model_obj.get('meta', {})
    else:
        pipe = model_obj['pipeline']
        proba = pipe.predict_proba(X)
        meta = model_obj.get('meta', {})

    # 阈值
    if args.thr_json:
        with open(args.thr_json, 'r', encoding='utf-8') as f:
            thr_obj = json.load(f)
        thr = float(thr_obj.get('leak_threshold', args.threshold))
    else:
        thr = float(args.threshold)

    pred = np.argmax(proba, axis=1)
    leak_mask = proba[:, LEAK_CLASS] >= thr
    pred[leak_mask] = LEAK_CLASS

    timeline = feat[['timestamp']].copy()
    timeline['pred'] = pred
    timeline['p_leak'] = proba[:, LEAK_CLASS]
    if 'device' in df.columns:
        timeline['device'] = df['device'].astype(str).values[:len(timeline)]

    events = postprocess_events(timeline, min_len=args.min_steps, gap_merge=args.gap_merge)

    if args.timeline:
        timeline.to_csv(args.timeline, index=False)
    if args.save:
        events.to_csv(args.save, index=False)

    print("Inference Done.")
    print(f"Model meta: {meta}")
    if args.save:
        print(f"Events saved: {args.save} | count={len(events)}")
    if args.timeline:
        print(f"Timeline saved: {args.timeline}")
    print("\n[Timeline head]\n", timeline.head().to_string(index=False))
    print("\n[Events]\n", events.to_string(index=False))

# ===================== CLI =====================
def build_parser():
    p = argparse.ArgumentParser(description="Leak Detection - Integrated (with liquidtest_sklearn rules)")
    sub = p.add_subparsers(dest='cmd')

    # 事件生成
    p_ev = sub.add_parser('events', help='Generate labeled_events.csv from raw water levels')
    p_ev.add_argument('--input', required=True, help='原始水位CSV/Parquet（含 code/device, msgTime/timestamp, liquidLevel_clean/level_mm）')
    p_ev.add_argument('--output', required=True, help='生成的 labeled_events.csv 路径')
    p_ev.add_argument('--freq', type=str, default='1H', help='等频重采样频率')
    p_ev.add_argument('--rise_th', type=float, default=RISE_TH, help='夜间急升阈值 mm/h')
    p_ev.add_argument('--drop_th', type=float, default=DROP_TH, help='每小时下降阈值 mm/h')
    p_ev.add_argument('--drop_sum_th', type=float, default=DROP_SUM_TH, help='累计下降阈值 mm')
    p_ev.add_argument('--drain_len_h', type=int, default=DRAIN_LEN_H, help='连续下降小时数')
    p_ev.add_argument('--spike_th', type=float, default=SPIKE_TH, help='突变过滤阈值 mm')
    p_ev.add_argument('--post_anom_block_h', type=int, default=POST_ANOMALY_BLOCK_H, help='异常后屏蔽排水小时数')

    # 训练
    p_tr = sub.add_parser('train', help='Train with GroupKFold; can auto-generate events')
    p_tr.add_argument('--input', required=True, help='训练输入CSV/Parquet（原始水位或已贴label的逐时刻表）')
    p_tr.add_argument('--events', default='', help='labeled_events.csv（如不提供可用 --gen_events 自动生成）')
    p_tr.add_argument('--gen_events', action='store_true', help='基于 liquidtest_sklearn 规则自动生成事件')
    p_tr.add_argument('--n_splits', type=int, default=5, help='GroupKFold折数')
    p_tr.add_argument('--freq', type=str, default='1H', help='等频重采样频率（用于 --use_raw_level 与事件生成）')
    p_tr.add_argument('--use_raw_level', action='store_true', help='基于 level_mm 自动构造特征（推荐）')
    # 事件生成阈值（可覆盖默认）
    p_tr.add_argument('--rise_th', type=float, default=RISE_TH, help='夜间急升阈值 mm/h')
    p_tr.add_argument('--drop_th', type=float, default=DROP_TH, help='每小时下降阈值 mm/h')
    p_tr.add_argument('--drop_sum_th', type=float, default=DROP_SUM_TH, help='累计下降阈值 mm')
    p_tr.add_argument('--drain_len_h', type=int, default=DRAIN_LEN_H, help='连续下降小时数')
    p_tr.add_argument('--spike_th', type=float, default=SPIKE_TH, help='突变过滤阈值 mm')
    p_tr.add_argument('--post_anom_block_h', type=int, default=POST_ANOMALY_BLOCK_H, help='异常后屏蔽排水小时数')

    # 推理
    p_in = sub.add_parser('infer', help='Run inference on new device data')
    p_in.add_argument('--input', required=True, help='推理输入CSV/Parquet（至少 timestamp,device,level_mm 或完整特征）')
    p_in.add_argument('--model', required=True, help='best_pipeline.pkl 或 ensemble_soft_voting.pkl')
    p_in.add_argument('--thr_json', default='', help='best_thresholds.json 路径（若缺省则使用 --threshold）')
    p_in.add_argument('--threshold', type=float, default=0.5, help='漏水概率阈值')
    p_in.add_argument('--freq', type=str, default='1H', help='特征构造频率')
    p_in.add_argument('--use_raw_level', action='store_true', help='若导出模型声明使用raw level造特征，需设置')
    p_in.add_argument('--min_steps', type=int, default=3, help='事件的最小持续步数')
    p_in.add_argument('--gap_merge', type=int, default=1, help='相邻片段合并的最大间隔步数')
    p_in.add_argument('--save', default='', help='保存事件CSV路径')
    p_in.add_argument('--timeline', default='', help='保存逐时刻预测CSV路径')

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == 'events':
        # 读取原始水位
        if args.input.lower().endswith(".parquet"):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
        # 兼容列名
        if 'code' not in df.columns and 'device' in df.columns:
            df = df.rename(columns={'device':'code'})
        if 'msgTime' not in df.columns and 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp':'msgTime'})
        if 'liquidLevel_clean' not in df.columns and 'level_mm' in df.columns:
            df = df.rename(columns={'level_mm':'liquidLevel_clean'})
        events = build_events_from_raw(
            df_raw=df, code_col='code', time_col='msgTime', level_col='liquidLevel_clean',
            freq=args.freq, rise_th=args.rise_th, drop_th=args.drop_th,
            drop_sum_th=args.drop_sum_th, drain_len_h=args.drain_len_h,
            spike_th=args.spike_th, post_anom_block_h=args.post_anom_block_h
        )
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        events.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"✅ 事件已生成：{args.output}（共{len(events)}条）")
    elif args.cmd == 'train':
        train_main(args)
    elif args.cmd == 'infer':
        infer_main(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
