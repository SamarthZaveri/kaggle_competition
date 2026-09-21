"""
Final Push: Target Encoding + MLP Stacking + Multi-Seed Averaging
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\test.csv"
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\submission_v2.csv"

TARGET = "addicted_label"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
SEEDS = [42, 101, 2024, 7, 999]   # multi-seed averaging

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")

BEST_XGB_PARAMS = {
    "n_estimators": 966, "max_depth": 4, "learning_rate": 0.0968039741820794,
    "subsample": 0.9804646024057977, "colsample_bytree": 0.6050391815827767,
    "min_child_weight": 11, "reg_alpha": 2.6175206276230206e-07,
    "reg_lambda": 3.0448897592728524e-08, "gamma": 0.3375131050010793,
}

PREDICTIVE_NUMERIC = ["daily_screen_time_hours", "weekend_screen_time",
                       "social_media_hours", "work_study_hours", "gaming_hours"]
NAIVE_NUMERIC = ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"]
NAIVE_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


# --------------------------------------------------------------------------- #
# Imputation (same as before)
# --------------------------------------------------------------------------- #
def naive_fill_fit(df, numeric_cols, categorical_cols):
    fills = {}
    for c in numeric_cols:
        fills[c] = df[c].median()
    for c in categorical_cols:
        mode = df[c].mode(dropna=True)
        fills[c] = mode.iloc[0] if len(mode) else "missing"
    return fills


def naive_fill_apply(df, fills):
    df = df.copy()
    for c, val in fills.items():
        df[c] = df[c].fillna(val)
    return df


def fit_predictive_imputer(train_df, col, feature_pool_train):
    known_mask = train_df[col].notna()
    feature_cols = [c for c in feature_pool_train.columns if c != col]
    X_known = feature_pool_train.loc[known_mask, feature_cols]
    y_known = train_df.loc[known_mask, col]
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, **XGB_GPU_BASE)
    model.fit(X_known, y_known)
    return model, np.sort(y_known.values)


def apply_predictive_imputer(df, col, feature_pool, model, sorted_known_values):
    missing_mask = df[col].isna()
    if missing_mask.sum() == 0:
        return df[col]
    feature_cols = [c for c in feature_pool.columns if c != col]
    X_missing = feature_pool.loc[missing_mask, feature_cols]
    raw_preds = model.predict(X_missing)
    idx = np.clip(np.searchsorted(sorted_known_values, raw_preds), 0, len(sorted_known_values) - 1)
    idx_left = np.clip(idx - 1, 0, len(sorted_known_values) - 1)
    left_dist = np.abs(sorted_known_values[idx_left] - raw_preds)
    right_dist = np.abs(sorted_known_values[idx] - raw_preds)
    final_idx = np.where(left_dist < right_dist, idx_left, idx)
    result = df[col].copy()
    result.loc[missing_mask] = sorted_known_values[final_idx]
    return result


# --------------------------------------------------------------------------- #
# NEW: leak-free target encoding
# --------------------------------------------------------------------------- #
def target_encode_train_oof(train_df, col, target_col, n_splits=5, smoothing=10, seed=RANDOM_STATE):
    """Out-of-fold target encoding for TRAIN (avoids leakage)."""
    global_mean = train_df[target_col].mean()
    oof_encoded = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(train_df, train_df[target_col]):
        fold_stats = train_df.iloc[tr_idx].groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (fold_stats["mean"] * fold_stats["count"] + global_mean * smoothing) / (
            fold_stats["count"] + smoothing
        )
        oof_encoded[va_idx] = train_df.iloc[va_idx][col].map(smoothed).fillna(global_mean).values
    return oof_encoded


def target_encode_fit_full(train_df, col, target_col, smoothing=10):
    """Full-data encoding map, used to encode TEST (no leakage risk since test has no target)."""
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return smoothed, global_mean


def apply_target_encoding(df, col, encoding_map, global_mean):
    return df[col].map(encoding_map).fillna(global_mean).values


# --------------------------------------------------------------------------- #
# Feature engineering (same as before, plus target-encoded columns added later)
# --------------------------------------------------------------------------- #
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

    comp_norm = (df["total_signal_composite"] - df["total_signal_composite"].min()) / (
        df["total_signal_composite"].max() - df["total_signal_composite"].min() + 1e-9
    )
    comp_norm = comp_norm.clip(0.001, 0.999)
    df["composite_logit"] = np.log(comp_norm / (1 - comp_norm))

    df["age_bucket"] = pd.cut(df["age"], bins=[17, 21, 25, 29, 33, 36],
                               labels=[0, 1, 2, 3, 4]).astype(float)
    df["age_bucket_x_composite"] = df["age_bucket"] * df["total_signal_composite"]

    df["notif_x_appopen"] = df["notifications_per_day"] * df["app_opens_per_day"]
    df["notif_per_appopen"] = (df["notifications_per_day"] /
                                df["app_opens_per_day"].replace(0, np.nan)).fillna(0)

    df["gender_x_stress"] = df["gender"].astype(str) + "_" + df["stress_level"].astype(str)
    df["academic_x_work"] = (df["academic_work_impact"].astype(str) + "_" +
                              pd.cut(df["work_study_hours"], bins=4).astype(str))

    df["sleep_hours_sq"] = df["sleep_hours"] ** 2
    df["notifications_sq"] = df["notifications_per_day"] ** 2
    return df


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")

    # ---------------- Imputation ----------------
    all_numeric = PREDICTIVE_NUMERIC + NAIVE_NUMERIC
    naive_fills = naive_fill_fit(train, all_numeric, NAIVE_CATEGORICAL)
    train_pool = naive_fill_apply(train, naive_fills)
    test_pool = naive_fill_apply(test, naive_fills)
    for c in NAIVE_CATEGORICAL:
        le = LabelEncoder().fit(train_pool[c].astype(str))
        train_pool[c] = le.transform(train_pool[c].astype(str))
        test_pool[c] = test_pool[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)
    train_pool = train_pool.drop(columns=[TARGET, ID_COL], errors="ignore")
    test_pool = test_pool.drop(columns=[ID_COL], errors="ignore")

    print("\nPredictive imputation:")
    train_imputed, test_imputed = train.copy(), test.copy()
    for col in PREDICTIVE_NUMERIC:
        model, sorted_vals = fit_predictive_imputer(train, col, train_pool)
        train_imputed[col] = apply_predictive_imputer(train, col, train_pool, model, sorted_vals)
        test_imputed[col] = apply_predictive_imputer(test, col, test_pool, model, sorted_vals)
        print(f"  {col}: done")

    train_imputed = naive_fill_apply(train_imputed, naive_fills)
    test_imputed = naive_fill_apply(test_imputed, naive_fills)

    # ---------------- Feature engineering ----------------
    train_final = engineer_features(train_imputed)
    test_final = engineer_features(test_imputed)

    # ---------------- NEW: target encoding for the 3 weak categoricals ----------------
    print("\nTarget encoding (leak-free OOF for train, full-fit for test):")
    for col in NAIVE_CATEGORICAL:
        train_final[f"{col}_te"] = target_encode_train_oof(train_final, col, TARGET)
        enc_map, gmean = target_encode_fit_full(train_final, col, TARGET)
        test_final[f"{col}_te"] = apply_target_encoding(test_final, col, enc_map, gmean)
        print(f"  {col}_te: done")

    # ---------------- Encode remaining categoricals (label encode, keep as backup features) ----------------
    feature_cols = [c for c in train_final.columns if c not in (ID_COL, TARGET)]
    X_train = train_final[feature_cols].copy()
    X_test = test_final[feature_cols].copy()
    y_train = train_final[TARGET].values

    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            le = LabelEncoder().fit(X_train[c].astype(str))
            X_train[c] = le.transform(X_train[c].astype(str))
            X_test[c] = X_test[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- OOF for stacking (XGB + CatBoost + MLP) ----------------
    print("\nGenerating OOF predictions (XGB + CatBoost + MLP) for stacking...")
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = np.zeros(len(y_train))
    oof_cat = np.zeros(len(y_train))
    oof_mlp = np.zeros(len(y_train))
    xgb_params = {**BEST_XGB_PARAMS, **XGB_GPU_BASE, "random_state": RANDOM_STATE}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        m1 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xgb_params)
        m1.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_xgb[va_idx] = m1.predict_proba(X_train.iloc[va_idx])[:, 1]

        m2 = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                                 task_type="GPU", devices="0", eval_metric="AUC",
                                 random_seed=RANDOM_STATE, verbose=False)
        m2.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_cat[va_idx] = m2.predict_proba(X_train.iloc[va_idx])[:, 1]

        m3 = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", alpha=1e-4,
                            learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=10,
                            max_iter=200, random_state=RANDOM_STATE)
        m3.fit(X_train_scaled[tr_idx], y_train[tr_idx])
        oof_mlp[va_idx] = m3.predict_proba(X_train_scaled[va_idx])[:, 1]

        print(f"  Fold {fold+1}: XGB={roc_auc_score(y_train[va_idx], oof_xgb[va_idx]):.4f}  "
              f"Cat={roc_auc_score(y_train[va_idx], oof_cat[va_idx]):.4f}  "
              f"MLP={roc_auc_score(y_train[va_idx], oof_mlp[va_idx]):.4f}")

    print(f"\nXGB OOF AUC: {roc_auc_score(y_train, oof_xgb):.4f}")
    print(f"Cat OOF AUC: {roc_auc_score(y_train, oof_cat):.4f}")
    print(f"MLP OOF AUC: {roc_auc_score(y_train, oof_mlp):.4f}")

    meta = LogisticRegression(max_iter=1000)
    stack_features_train = np.column_stack([oof_xgb, oof_cat, oof_mlp])
    meta.fit(stack_features_train, y_train)
    stack_oof = meta.predict_proba(stack_features_train)[:, 1]
    print(f"Stacked (XGB+Cat+MLP) OOF AUC: {roc_auc_score(y_train, stack_oof):.4f}")

    # ---------------- Multi-seed averaging on final refit ----------------
    print(f"\nRefitting final models across {len(SEEDS)} seeds on 100% of train...")
    test_pred_xgb_seeds = []
    test_pred_cat_seeds = []
    test_pred_mlp_seeds = []

    for seed in SEEDS:
        xgb_seed_params = {**BEST_XGB_PARAMS, **XGB_GPU_BASE, "random_state": seed}
        m1 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xgb_seed_params)
        m1.fit(X_train, y_train)
        test_pred_xgb_seeds.append(m1.predict_proba(X_test)[:, 1])

        m2 = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
                                 task_type="GPU", devices="0", eval_metric="AUC",
                                 random_seed=seed, verbose=False)
        m2.fit(X_train, y_train)
        test_pred_cat_seeds.append(m2.predict_proba(X_test)[:, 1])

        m3 = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", alpha=1e-4,
                            learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=10,
                            max_iter=200, random_state=seed)
        m3.fit(X_train_scaled, y_train)
        test_pred_mlp_seeds.append(m3.predict_proba(X_test_scaled)[:, 1])

        print(f"  Seed {seed}: done")

    test_pred_xgb = np.mean(test_pred_xgb_seeds, axis=0)
    test_pred_cat = np.mean(test_pred_cat_seeds, axis=0)
    test_pred_mlp = np.mean(test_pred_mlp_seeds, axis=0)

    stack_features_test = np.column_stack([test_pred_xgb, test_pred_cat, test_pred_mlp])
    test_pred_final = meta.predict_proba(stack_features_test)[:, 1]

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved submission -> {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()