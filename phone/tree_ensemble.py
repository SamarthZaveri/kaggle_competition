"""
Tree-Family Ensemble: RandomForest + LightGBM + CatBoost + XGBoost
======================================================================
Plus an Isolation Forest anomaly score as an engineered feature.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\test.csv"
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\submission_treefamily.csv"

TARGET = "addicted_label"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
STRONG_FEATS = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]


# --------------------------------------------------------------------------- #
# Imputation + feature engineering (unchanged)
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

    df["age_bucket"] = pd.cut(df["age"], bins=[17, 21, 25, 29, 33, 36], labels=[0, 1, 2, 3, 4]).astype(float)
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


def add_isolation_forest_feature(train_df, test_df):
    """Anomaly score on the strong numeric features -- fed in as ONE extra
    feature to the tree models, not used as a standalone classifier."""
    iso = IsolationForest(n_estimators=200, contamination="auto",
                           random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(train_df[STRONG_FEATS])
    train_score = iso.score_samples(train_df[STRONG_FEATS])
    test_score = iso.score_samples(test_df[STRONG_FEATS])
    return train_score, test_score


# --------------------------------------------------------------------------- #
# Optuna tuning for LightGBM (RF uses sane defaults, not tuned -- diminishing
# returns on an already-weaker-family model)
# --------------------------------------------------------------------------- #
def tune_lightgbm(X, y, n_trials=25):
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=150_000, stratify=y, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=RANDOM_STATE, verbosity=-1,
        )
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, va_idx in kf.split(X_sub, y_sub):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_sub.iloc[tr_idx], y_sub.iloc[tr_idx])
            preds = model.predict_proba(X_sub.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_sub.iloc[va_idx], preds))
        return np.mean(scores)

    print(f"\nTuning LightGBM ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best LightGBM subsample AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train_tree_family_ensemble(X_train, y_train, X_test, best_lgb_params):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y_train)
    oof = {k: np.zeros(n) for k in ["xgb", "cat", "lgb", "rf"]}
    test_preds = {k: [] for k in oof}

    xgb_params = {**BEST_XGB_PARAMS, **XGB_GPU_BASE, "random_state": RANDOM_STATE}
    cat_params = {**BEST_CAT_PARAMS, "task_type": "GPU", "devices": "0",
                  "random_seed": RANDOM_STATE, "verbose": False}
    lgb_params = {**best_lgb_params, "random_state": RANDOM_STATE, "verbosity": -1}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xgb_params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["xgb"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["xgb"].append(m.predict_proba(X_test)[:, 1])

        m = CatBoostClassifier(**cat_params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["cat"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["cat"].append(m.predict_proba(X_test)[:, 1])

        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["lgb"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["lgb"].append(m.predict_proba(X_test)[:, 1])

        m = RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=5,
                                    n_jobs=-1, random_state=RANDOM_STATE)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["rf"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["rf"].append(m.predict_proba(X_test)[:, 1])

        print(f"Fold {fold+1}: " + "  ".join(
            f"{k}={roc_auc_score(y_train[va_idx], oof[k][va_idx]):.4f}" for k in oof
        ))

    print("\n=== Overall OOF AUC per model ===")
    for k in oof:
        print(f"  {k}: {roc_auc_score(y_train, oof[k]):.4f}")

    test_pred_avg = {k: np.mean(test_preds[k], axis=0) for k in test_preds}
    return oof, test_pred_avg


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")

    train_imputed, test_imputed = run_imputation(train, test)
    train_final = engineer_features(train_imputed)
    test_final = engineer_features(test_imputed)

    print("\nAdding Isolation Forest anomaly score feature...")
    train_iso, test_iso = add_isolation_forest_feature(train_final, test_final)
    train_final["iso_forest_score"] = train_iso
    test_final["iso_forest_score"] = test_iso

    feature_cols = [c for c in train_final.columns if c not in (ID_COL, TARGET)]
    X_train = train_final[feature_cols].copy()
    X_test = test_final[feature_cols].copy()
    y_train = train_final[TARGET].values

    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            le = LabelEncoder().fit(X_train[c].astype(str))
            X_train[c] = le.transform(X_train[c].astype(str))
            X_test[c] = X_test[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

    y_train_series = pd.Series(y_train)
    best_lgb_params = tune_lightgbm(X_train, y_train_series, n_trials=25)

    print("\nTraining tree-family ensemble (XGB, CatBoost, LightGBM, RandomForest)...")
    oof, test_preds = train_tree_family_ensemble(X_train, y_train, X_test, best_lgb_params)

    meta = LogisticRegression(max_iter=1000)
    model_names = list(oof.keys())
    stack_train = np.column_stack([oof[k] for k in model_names])
    stack_test = np.column_stack([test_preds[k] for k in model_names])

    kf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y_train))
    for tr_idx, va_idx in kf2.split(stack_train, y_train):
        meta_fold = LogisticRegression(max_iter=1000)
        meta_fold.fit(stack_train[tr_idx], y_train[tr_idx])
        stack_oof[va_idx] = meta_fold.predict_proba(stack_train[va_idx])[:, 1]

    stack_auc = roc_auc_score(y_train, stack_oof)
    print(f"\nStacked (tree-family) OOF AUC: {stack_auc:.4f}")

    meta.fit(stack_train, y_train)
    test_final_pred = meta.predict_proba(stack_test)[:, 1]

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_final_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved -> {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()