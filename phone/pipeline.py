"""
Bagged Ensemble: 8x-seeded XGBoost + 8x-seeded CatBoost, averaged then stacked
=================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\test.csv"
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\submission_bagged.csv"

TARGET = "addicted_label"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
N_BAGS = 8   # models per fold per model-type -- raise to 12-15 if you have time

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")

BEST_XGB_PARAMS = {
    "n_estimators": 966, "max_depth": 4, "learning_rate": 0.0968039741820794,
    "subsample": 0.9804646024057977, "colsample_bytree": 0.6050391815827767,
    "min_child_weight": 11, "reg_alpha": 2.6175206276230206e-07,
    "reg_lambda": 3.0448897592728524e-08, "gamma": 0.3375131050010793,
}
BEST_CAT_PARAMS = {
    "iterations": 838, "depth": 5, "learning_rate": 0.09975733610372729,
    "l2_leaf_reg": 11.129953671946815,
}

PREDICTIVE_NUMERIC = ["daily_screen_time_hours", "weekend_screen_time",
                       "social_media_hours", "work_study_hours", "gaming_hours"]
NAIVE_NUMERIC = ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"]
NAIVE_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]

rng = np.random.default_rng(RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Imputation + feature engineering (unchanged from your working pipeline)
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


def run_imputation(train, test):
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

    print("Predictive imputation:")
    train_imputed, test_imputed = train.copy(), test.copy()
    for col in PREDICTIVE_NUMERIC:
        model, sorted_vals = fit_predictive_imputer(train, col, train_pool)
        train_imputed[col] = apply_predictive_imputer(train, col, train_pool, model, sorted_vals)
        test_imputed[col] = apply_predictive_imputer(test, col, test_pool, model, sorted_vals)
        print(f"  {col}: done")

    train_imputed = naive_fill_apply(train_imputed, naive_fills)
    test_imputed = naive_fill_apply(test_imputed, naive_fills)
    return train_imputed, test_imputed


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


# --------------------------------------------------------------------------- #
# Bagged training
# --------------------------------------------------------------------------- #
def jittered_xgb_params(base_params, seed):
    p = dict(base_params)
    p["subsample"] = float(np.clip(base_params["subsample"] + rng.uniform(-0.1, 0.1), 0.5, 1.0))
    p["colsample_bytree"] = float(np.clip(base_params["colsample_bytree"] + rng.uniform(-0.1, 0.1), 0.4, 1.0))
    p["random_state"] = seed
    return p


def jittered_cat_params(base_params, seed):
    p = dict(base_params)
    p["random_seed"] = seed
    return p


def train_bagged_ensemble(X_train, y_train, X_test):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y_train)
    oof_xgb_bagged = np.zeros(n)
    oof_cat_bagged = np.zeros(n)
    test_xgb_all_folds = []
    test_cat_all_folds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        xgb_bag_preds_va = np.zeros(len(va_idx))
        cat_bag_preds_va = np.zeros(len(va_idx))
        xgb_bag_preds_test = np.zeros(len(X_test))
        cat_bag_preds_test = np.zeros(len(X_test))

        for bag in range(N_BAGS):
            seed = RANDOM_STATE * 1000 + fold * 100 + bag
            xp = jittered_xgb_params(BEST_XGB_PARAMS, seed)
            m1 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xp, **XGB_GPU_BASE)
            m1.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            xgb_bag_preds_va += m1.predict_proba(X_train.iloc[va_idx])[:, 1] / N_BAGS
            xgb_bag_preds_test += m1.predict_proba(X_test)[:, 1] / (N_BAGS * N_SPLITS)

            cp = jittered_cat_params(BEST_CAT_PARAMS, seed)
            m2 = CatBoostClassifier(**cp, task_type="GPU", devices="0",
                                     eval_metric="AUC", verbose=False)
            m2.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            cat_bag_preds_va += m2.predict_proba(X_train.iloc[va_idx])[:, 1] / N_BAGS
            cat_bag_preds_test += m2.predict_proba(X_test)[:, 1] / (N_BAGS * N_SPLITS)

            print(f"  bag {bag+1}/{N_BAGS} done")

        oof_xgb_bagged[va_idx] = xgb_bag_preds_va
        oof_cat_bagged[va_idx] = cat_bag_preds_va
        test_xgb_all_folds.append(xgb_bag_preds_test)
        test_cat_all_folds.append(cat_bag_preds_test)

        print(f"  Fold {fold+1} bagged: XGB={roc_auc_score(y_train[va_idx], oof_xgb_bagged[va_idx]):.4f}  "
              f"Cat={roc_auc_score(y_train[va_idx], oof_cat_bagged[va_idx]):.4f}")

    print(f"\nBagged XGBoost OOF AUC: {roc_auc_score(y_train, oof_xgb_bagged):.4f}")
    print(f"Bagged CatBoost OOF AUC: {roc_auc_score(y_train, oof_cat_bagged):.4f}")

    test_xgb_final = np.sum(test_xgb_all_folds, axis=0)
    test_cat_final = np.sum(test_cat_all_folds, axis=0)

    return oof_xgb_bagged, oof_cat_bagged, test_xgb_final, test_cat_final


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")
    print(f"Bagging config: {N_BAGS} models/fold/type x {N_SPLITS} folds x 2 model types "
          f"= {N_BAGS*N_SPLITS*2} total fits")

    train_imputed, test_imputed = run_imputation(train, test)
    train_final = engineer_features(train_imputed)
    test_final = engineer_features(test_imputed)

    feature_cols = [c for c in train_final.columns if c not in (ID_COL, TARGET)]
    X_train = train_final[feature_cols].copy()
    X_test = test_final[feature_cols].copy()
    y_train = train_final[TARGET].values

    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            le = LabelEncoder().fit(X_train[c].astype(str))
            X_train[c] = le.transform(X_train[c].astype(str))
            X_test[c] = X_test[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

    oof_xgb, oof_cat, test_xgb, test_cat = train_bagged_ensemble(X_train, y_train, X_test)

    meta = LogisticRegression(max_iter=1000)
    stack_train = np.column_stack([oof_xgb, oof_cat])
    meta.fit(stack_train, y_train)
    stack_oof = meta.predict_proba(stack_train)[:, 1]
    stack_auc = roc_auc_score(y_train, stack_oof)
    print(f"\nFinal stacked (bagged XGB + bagged CatBoost) OOF AUC: {stack_auc:.4f}")

    stack_test = np.column_stack([test_xgb, test_cat])
    test_final_pred = meta.predict_proba(stack_test)[:, 1]

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_final_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved -> {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()