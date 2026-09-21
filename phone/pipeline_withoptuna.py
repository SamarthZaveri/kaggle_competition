"""
Full Optimization Pass: Optuna-tuned XGBoost + expanded features + stacking
=============================================================================
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TARGET = "addicted_label"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

XGB_GPU_BASE = dict(tree_method="hist", device="cuda", random_state=RANDOM_STATE)

PREDICTIVE_NUMERIC = ["daily_screen_time_hours", "weekend_screen_time",
                       "social_media_hours", "work_study_hours", "gaming_hours"]
NAIVE_NUMERIC = ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"]
NAIVE_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Imputation (unchanged from before)
# --------------------------------------------------------------------------- #
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
                              subsample=0.8, colsample_bytree=0.8, **XGB_GPU_BASE)
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
    print(f"  {col}: imputed {missing_mask.sum()} values")
    return result


def run_imputation(df):
    df = df.copy()
    all_numeric = PREDICTIVE_NUMERIC + NAIVE_NUMERIC
    feature_pool = naive_fill(df, all_numeric, NAIVE_CATEGORICAL)
    for c in NAIVE_CATEGORICAL:
        feature_pool[c] = LabelEncoder().fit_transform(feature_pool[c].astype(str))
    feature_pool = feature_pool.drop(columns=[TARGET, ID_COL], errors="ignore")

    print("Predictive (quantile-matched) imputation:")
    for col in PREDICTIVE_NUMERIC:
        df[col] = quantile_match_impute(df, col, feature_pool)

    df = naive_fill(df, NAIVE_NUMERIC, NAIVE_CATEGORICAL)
    return df


# --------------------------------------------------------------------------- #
# Step 2: expanded feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df):
    df = df.copy()

    # original features
    df["weekend_vs_daily_diff"] = df["weekend_screen_time"] - df["daily_screen_time_hours"]
    df["social_media_share"] = (df["social_media_hours"] /
                                 df["daily_screen_time_hours"].replace(0, np.nan)).fillna(0)
    df["total_signal_composite"] = (df["daily_screen_time_hours"] +
                                     df["weekend_screen_time"] + df["social_media_hours"])
    df["screen_to_work_ratio"] = (df["daily_screen_time_hours"] /
                                   df["work_study_hours"].replace(0, np.nan)).fillna(0)
    df["composite_x_social_share"] = df["total_signal_composite"] * df["social_media_share"]
    df["composite_squared"] = df["total_signal_composite"] ** 2

    # NEW: logit-style transform on composite (sigmoid shape suggests this linearizes it)
    comp_norm = (df["total_signal_composite"] - df["total_signal_composite"].min()) / (
        df["total_signal_composite"].max() - df["total_signal_composite"].min() + 1e-9
    )
    comp_norm = comp_norm.clip(0.001, 0.999)
    df["composite_logit"] = np.log(comp_norm / (1 - comp_norm))

    # NEW: age buckets interacted with screen time
    df["age_bucket"] = pd.cut(df["age"], bins=[17, 21, 25, 29, 33, 36],
                               labels=[0, 1, 2, 3, 4]).astype(float)
    df["age_bucket_x_composite"] = df["age_bucket"] * df["total_signal_composite"]

    # NEW: notification/app-open interaction (both weak alone)
    df["notif_x_appopen"] = df["notifications_per_day"] * df["app_opens_per_day"]
    df["notif_per_appopen"] = (df["notifications_per_day"] /
                                df["app_opens_per_day"].replace(0, np.nan)).fillna(0)

    # NEW: categorical interactions
    df["gender_x_stress"] = df["gender"].astype(str) + "_" + df["stress_level"].astype(str)
    df["academic_x_work"] = (df["academic_work_impact"].astype(str) + "_" +
                              pd.cut(df["work_study_hours"], bins=4).astype(str))

    # NEW: polynomial on weak features (in case of non-linear signal)
    df["sleep_hours_sq"] = df["sleep_hours"] ** 2
    df["notifications_sq"] = df["notifications_per_day"] ** 2

    return df


# --------------------------------------------------------------------------- #
# Step 1: Optuna hyperparameter search for XGBoost
# --------------------------------------------------------------------------- #
def tune_xgboost(X, y, n_trials=40):
    # subsample for faster search, then retrain final model on full data
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=150_000, stratify=y, random_state=RANDOM_STATE
    )

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 15),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            **XGB_GPU_BASE,
        )
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, va_idx in kf.split(X_sub, y_sub):
            model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **params)
            model.fit(X_sub.iloc[tr_idx], y_sub.iloc[tr_idx])
            preds = model.predict_proba(X_sub.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_sub.iloc[va_idx], preds))
        return np.mean(scores)

    print(f"\nRunning Optuna search ({n_trials} trials on {len(X_sub)}-row subsample)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest CV AUC (subsample): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


# --------------------------------------------------------------------------- #
# Step 3+4: train tuned XGBoost + CatBoost + MLP, then STACK (not blend)
# --------------------------------------------------------------------------- #
def train_and_stack(df, best_xgb_params):
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    X = df[feature_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    y = df[TARGET].values

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))

    xgb_params = {**best_xgb_params, **XGB_GPU_BASE}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xgb_params)
        xgb_model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_xgb[va_idx] = xgb_model.predict_proba(X.iloc[va_idx])[:, 1]

        cat_model = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
            task_type="GPU", devices="0", eval_metric="AUC",
            random_seed=RANDOM_STATE, verbose=False,
        )
        cat_model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_cat[va_idx] = cat_model.predict_proba(X.iloc[va_idx])[:, 1]

        print(f"Fold {fold+1}: XGB={roc_auc_score(y[va_idx], oof_xgb[va_idx]):.4f}  "
              f"Cat={roc_auc_score(y[va_idx], oof_cat[va_idx]):.4f}")

    print(f"\nTuned XGBoost AUC: {roc_auc_score(y, oof_xgb):.4f}")
    print(f"CatBoost AUC:      {roc_auc_score(y, oof_cat):.4f}")

    # -------- STACKING: meta-model on the 2 OOF prediction columns --------
    stack_X = np.column_stack([oof_xgb, oof_cat])
    stack_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y))
    for tr_idx, va_idx in stack_kf.split(stack_X, y):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(stack_X[tr_idx], y[tr_idx])
        stack_oof[va_idx] = meta.predict_proba(stack_X[va_idx])[:, 1]

    stack_auc = roc_auc_score(y, stack_oof)
    print(f"\nStacked meta-model AUC: {stack_auc:.4f}")

    # compare to simple best-weight blend as a sanity check
    best_auc, best_w = 0, 1.0
    for w in np.arange(0, 1.01, 0.02):
        blend = w * oof_xgb + (1 - w) * oof_cat
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"Best linear blend AUC:  {best_auc:.4f}  (weight XGB={best_w:.2f}, CatBoost={1-best_w:.2f})")

    final_pred = stack_oof if stack_auc >= best_auc else (
        best_w * oof_xgb + (1 - best_w) * oof_cat
    )
    method = "stacking" if stack_auc >= best_auc else "blending"
    print(f"\n>>> Using {method} as final method, AUC = {max(stack_auc, best_auc):.4f}")

    best_acc, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        acc = accuracy_score(y, (final_pred >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"Best threshold: {best_t:.2f} -> Accuracy: {best_acc:.4f}")
    print(f"LogLoss: {log_loss(y, final_pred):.4f}")

def main():
    df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded: {df.shape}")

    df = run_imputation(df)
    df = engineer_features(df)
    print(f"\nTotal features after engineering: {df.shape[1] - 2}")

    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    X = df[feature_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    y = df[TARGET]

    best_params = tune_xgboost(X, y, n_trials=40)
    train_and_stack(df, best_params)


if __name__ == "__main__":
    main()