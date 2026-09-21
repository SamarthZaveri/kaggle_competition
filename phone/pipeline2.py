"""
XGBoost (GPU) + CatBoost (GPU) + sklearn MLP (CPU, no torch needed) Ensemble
==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TARGET = "addicted_label"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

XGB_GPU = dict(tree_method="hist", device="cuda", random_state=RANDOM_STATE)

PREDICTIVE_NUMERIC = ["daily_screen_time_hours", "weekend_screen_time",
                       "social_media_hours", "work_study_hours", "gaming_hours"]
NAIVE_NUMERIC = ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"]
NAIVE_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def naive_fill(df, numeric_cols, categorical_cols):
    df = df.copy()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in categorical_cols:
        mode = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "missing")
    return df


def quantile_match_impute(df, col, feature_pool):
    known_mask = df[col].notna()
    missing_mask = ~known_mask
    if missing_mask.sum() == 0:
        return df[col]

    feature_cols = [c for c in feature_pool.columns if c != col]
    X_known = feature_pool.loc[known_mask, feature_cols]
    y_known = df.loc[known_mask, col]

    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, **XGB_GPU)
    model.fit(X_known, y_known)

    X_missing = feature_pool.loc[missing_mask, feature_cols]
    raw_preds = model.predict(X_missing)

    sorted_known = np.sort(y_known.values)
    idx = np.clip(np.searchsorted(sorted_known, raw_preds), 0, len(sorted_known) - 1)
    idx_left = np.clip(idx - 1, 0, len(sorted_known) - 1)
    left_dist = np.abs(sorted_known[idx_left] - raw_preds)
    right_dist = np.abs(sorted_known[idx] - raw_preds)
    final_idx = np.where(left_dist < right_dist, idx_left, idx)
    matched_values = sorted_known[final_idx]

    result = df[col].copy()
    result.loc[missing_mask] = matched_values
    print(f"  {col}: imputed {missing_mask.sum()} values (quantile-matched, GPU)")
    return result


def run_imputation(df):
    df = df.copy()
    all_numeric = PREDICTIVE_NUMERIC + NAIVE_NUMERIC
    feature_pool = naive_fill(df, all_numeric, NAIVE_CATEGORICAL)
    for c in NAIVE_CATEGORICAL:
        feature_pool[c] = LabelEncoder().fit_transform(feature_pool[c].astype(str))
    feature_pool = feature_pool.drop(columns=[TARGET, ID_COL], errors="ignore")

    print("Predictive (quantile-matched) imputation on GPU:")
    for col in PREDICTIVE_NUMERIC:
        df[col] = quantile_match_impute(df, col, feature_pool)

    df = naive_fill(df, NAIVE_NUMERIC, NAIVE_CATEGORICAL)
    return df


def engineer_features(df):
    df = df.copy()
    df["weekend_vs_daily_diff"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    df["social_media_share"] = (df["social_media_hours"] /
                                 df["daily_screen_time_hours"].replace(0, np.nan)).fillna(0)
    df["total_signal_composite"] = (df["daily_screen_time_hours"] +
                                     df["weekend_screen_time"] + df["social_media_hours"])
    df["screen_to_work_ratio"] = (df["daily_screen_time_hours"] /
                                   df["work_study_hours"].replace(0, np.nan)).fillna(0)
    df["composite_x_social_share"] = df["total_signal_composite"] * df["social_media_share"]
    df["composite_squared"] = df["total_signal_composite"] ** 2
    return df


def train_ensemble(df):
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    X = df[feature_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    y = df[TARGET].values

    # MLP needs scaled features; tree models don't
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        # XGBoost (GPU)
        xgb_model = xgb.XGBClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="auc", **XGB_GPU,
        )
        xgb_model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_xgb[va_idx] = xgb_model.predict_proba(X.iloc[va_idx])[:, 1]

        # CatBoost (GPU)
        cat_model = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.03,
            l2_leaf_reg=3.0, task_type="GPU", devices="0",
            eval_metric="AUC", random_seed=RANDOM_STATE, verbose=False,
        )
        cat_model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_cat[va_idx] = cat_model.predict_proba(X.iloc[va_idx])[:, 1]

        # MLP (CPU, sklearn, no torch needed)
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
        mlp.fit(X_scaled[tr_idx], y[tr_idx])
        oof_mlp[va_idx] = mlp.predict_proba(X_scaled[va_idx])[:, 1]

        auc_xgb = roc_auc_score(y[va_idx], oof_xgb[va_idx])
        auc_cat = roc_auc_score(y[va_idx], oof_cat[va_idx])
        auc_mlp = roc_auc_score(y[va_idx], oof_mlp[va_idx])
        print(f"Fold {fold+1}: XGB={auc_xgb:.4f}  CatBoost={auc_cat:.4f}  MLP={auc_mlp:.4f}")

    print(f"\nXGBoost overall AUC:  {roc_auc_score(y, oof_xgb):.4f}")
    print(f"CatBoost overall AUC: {roc_auc_score(y, oof_cat):.4f}")
    print(f"MLP overall AUC:      {roc_auc_score(y, oof_mlp):.4f}")

    best_auc, best_weights = 0, (1, 0, 0)
    step = 0.05
    for w_xgb in np.arange(0, 1.01, step):
        for w_cat in np.arange(0, 1.01 - w_xgb, step):
            w_mlp = 1 - w_xgb - w_cat
            blend = w_xgb * oof_xgb + w_cat * oof_cat + w_mlp * oof_mlp
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc, best_weights = auc, (w_xgb, w_cat, w_mlp)

    w_xgb, w_cat, w_mlp = best_weights
    print(f"\nBest blend: {w_xgb:.2f}*XGB + {w_cat:.2f}*CatBoost + {w_mlp:.2f}*MLP")
    print(f"Blend AUC: {best_auc:.4f}")

    blend_final = w_xgb * oof_xgb + w_cat * oof_cat + w_mlp * oof_mlp
    best_acc, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        acc = accuracy_score(y, (blend_final >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"Best threshold: {best_t:.2f} -> Accuracy: {best_acc:.4f}")
    print(f"LogLoss: {log_loss(y, blend_final):.4f}")


def main():
    df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded: {df.shape}")
    df = run_imputation(df)
    df = engineer_features(df)
    train_ensemble(df)


if __name__ == "__main__":
    main()