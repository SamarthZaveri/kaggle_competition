"""
EV Purchase Prediction: Ensemble v2
======================================
Fixes applied vs v1 (which plateaued at 0.9421 stacked OOF AUC):

1. WOE / leak-free smoothed-target-encoded features for every categorical
   and binned-numeric driver, PLUS a master_logit_score that sums the
   logit-transformed encodings of the strongest drivers -- gives the
   models a near-direct read on the log-odds structure that (per the
   oracle_check.py diagnostic) a coarse group-mean can already partially
   recover, instead of making trees rebuild it from scratch via splits.

2. Tuning objective now averages AUC over a 3-fold split instead of a
   single 80/20 split. v1 had all three libraries independently converge
   to suspiciously shallow trees (depth 3) -- a strong sign the single
   validation split was noisy enough to mislead hyperparameter selection
   rather than shallow genuinely being best.

3. CatBoost depth search capped at 8 (GPU CatBoost errors above depth 8).

Run oracle_check.py first if you haven't -- its printed AUC ceiling per
key-combination tells you roughly what to expect here before this runs.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\train (2).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\test (1).csv"
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\submission_ensemble_v2.csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
TUNE_CV_SPLITS = 3      # robust CV-averaged objective, instead of 1 noisy split
N_TRIALS_XGB = 25
N_TRIALS_LGB = 25
N_TRIALS_CAT = 25
WOE_SMOOTHING = 50

RAW_NUMERIC = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
               "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
               "Environmental_Concern_Level"]
RAW_CATEGORICAL = ["Gender", "City_Type", "Current_Car_Type",
                    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")


# --------------------------------------------------------------------------- #
# Feature engineering (same interaction features as v1)
# --------------------------------------------------------------------------- #
def engineer_features(df):
    df = df.copy()
    df["total_charging_access"] = df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    df["charging_home_work_ratio"] = df["Charging_Stations_Near_Home"] / (df["Charging_Stations_Near_Work"] + 1)
    df["income_x_env_concern"] = df["Annual_Income_USD"] * df["Environmental_Concern_Level"]
    df["income_per_env_concern"] = df["Annual_Income_USD"] / (df["Environmental_Concern_Level"] + 1)
    df["commute_x_env_concern"] = df["Daily_Commute_km"] * df["Environmental_Concern_Level"]
    df["subsidy_x_range_anxiety"] = df["Subsidy_Available"].astype(str) + "_" + df["Range_Anxiety_Level"].astype(str)
    df["subsidy_x_home_charging"] = df["Subsidy_Available"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)
    df["home_charging_x_range_anxiety"] = df["Home_Charging_Possible"].astype(str) + "_" + df["Range_Anxiety_Level"].astype(str)
    df["city_x_home_charging"] = df["City_Type"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)
    return df


ENGINEERED_NUMERIC = ["total_charging_access", "charging_home_work_ratio",
                       "income_x_env_concern", "income_per_env_concern", "commute_x_env_concern"]
ENGINEERED_CATEGORICAL = ["subsidy_x_range_anxiety", "subsidy_x_home_charging",
                           "home_charging_x_range_anxiety", "city_x_home_charging"]


# --------------------------------------------------------------------------- #
# Binning (fit edges on train only, apply to both train/test consistently)
# --------------------------------------------------------------------------- #
def fit_bin_edges(series, q=5):
    _, edges = pd.qcut(series, q=q, duplicates="drop", retbins=True)
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bins(series, edges):
    return pd.cut(series, bins=edges, labels=False).astype(str)


# --------------------------------------------------------------------------- #
# Leak-free (K-fold OOF) smoothed target / WOE encoding
# --------------------------------------------------------------------------- #
def oof_smoothed_encode(train_key, y, test_key, n_splits=N_SPLITS, smoothing=WOE_SMOOTHING, seed=RANDOM_STATE):
    oof = np.zeros(len(train_key))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, va_idx in kf.split(train_key, y):
        tr_k, tr_y = train_key.iloc[tr_idx], y[tr_idx]
        global_rate = tr_y.mean()
        stats = pd.DataFrame({"key": tr_k, "y": tr_y}).groupby("key")["y"].agg(["sum", "count"])
        stats["rate"] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
        oof[va_idx] = train_key.iloc[va_idx].map(stats["rate"]).fillna(global_rate).values

    # full-train fit for test set
    global_rate = y.mean()
    stats = pd.DataFrame({"key": train_key, "y": y}).groupby("key")["y"].agg(["sum", "count"])
    stats["rate"] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    test_enc = test_key.map(stats["rate"]).fillna(global_rate).values

    return oof, test_enc


def to_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


WOE_SINGLE_KEYS = ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
                    "Home_Charging_Possible", "City_Type", "Current_Car_Type", "Gender",
                    "income_bin", "age_bin", "commute_bin"]
WOE_COMBO_KEYS = [
    ["Subsidy_Available", "Range_Anxiety_Level"],
    ["Subsidy_Available", "Home_Charging_Possible"],
    ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level", "Home_Charging_Possible"],
    ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level", "Home_Charging_Possible", "income_bin"],
]
MASTER_SCORE_KEYS = ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
                      "Home_Charging_Possible", "income_bin"]


def build_woe_features(train_df, test_df, y):
    train_out = pd.DataFrame(index=train_df.index)
    test_out = pd.DataFrame(index=test_df.index)

    income_edges = fit_bin_edges(train_df["Annual_Income_USD"], 5)
    age_edges = fit_bin_edges(train_df["Age"], 5)
    commute_edges = fit_bin_edges(train_df["Daily_Commute_km"], 5)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["income_bin"] = apply_bins(train_df["Annual_Income_USD"], income_edges)
    test_df["income_bin"] = apply_bins(test_df["Annual_Income_USD"], income_edges)
    train_df["age_bin"] = apply_bins(train_df["Age"], age_edges)
    test_df["age_bin"] = apply_bins(test_df["Age"], age_edges)
    train_df["commute_bin"] = apply_bins(train_df["Daily_Commute_km"], commute_edges)
    test_df["commute_bin"] = apply_bins(test_df["Daily_Commute_km"], commute_edges)

    master_logit_train = np.zeros(len(train_df))
    master_logit_test = np.zeros(len(test_df))

    for k in WOE_SINGLE_KEYS:
        train_key = train_df[k].astype(str)
        test_key = test_df[k].astype(str)
        oof, test_enc = oof_smoothed_encode(train_key, y, test_key)
        col = f"woe_{k}"
        train_out[col] = oof
        test_out[col] = test_enc
        if k in MASTER_SCORE_KEYS:
            master_logit_train += to_logit(oof)
            master_logit_test += to_logit(test_enc)

    for combo in WOE_COMBO_KEYS:
        name = "woe_" + "_x_".join(combo)
        train_key = train_df[combo].astype(str).agg("_".join, axis=1)
        test_key = test_df[combo].astype(str).agg("_".join, axis=1)
        oof, test_enc = oof_smoothed_encode(train_key, y, test_key)
        train_out[name] = oof
        test_out[name] = test_enc

    train_out["master_logit_score"] = master_logit_train
    test_out["master_logit_score"] = master_logit_test

    return train_out, test_out


# --------------------------------------------------------------------------- #
# Encoding for raw + engineered categoricals (label encoding for XGB/LGB;
# CatBoost gets raw strings natively)
# --------------------------------------------------------------------------- #
def build_label_encoders(train_df, cat_cols):
    return {c: LabelEncoder().fit(train_df[c].astype(str)) for c in cat_cols}


def apply_label_encoders(df, cat_cols, encoders):
    out = df.copy()
    for c in cat_cols:
        le = encoders[c]
        out[c] = out[c].astype(str).map(lambda v: (le.transform([v])[0] if v in le.classes_ else -1))
    return out


# --------------------------------------------------------------------------- #
# Robust (3-fold CV-averaged) Optuna tuning
# --------------------------------------------------------------------------- #
def cv_auc_xgb(X, y, params, n_splits=TUNE_CV_SPLITS):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in kf.split(X, y):
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                   random_state=RANDOM_STATE, early_stopping_rounds=50,
                                   **params, **XGB_GPU_BASE)
        model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
        scores.append(roc_auc_score(y[va_idx], model.predict_proba(X.iloc[va_idx])[:, 1]))
    return float(np.mean(scores))


def tune_xgb(X, y, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
        return cv_auc_xgb(X, y, params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best XGB CV AUC: {study.best_value:.4f}")
    print(f"Best XGB params: {study.best_params}")
    return study.best_params


def cv_auc_lgb(X, y, params, n_splits=TUNE_CV_SPLITS):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in kf.split(X, y):
        model = lgb.LGBMClassifier(objective="binary", metric="auc",
                                    random_state=RANDOM_STATE, verbosity=-1, **params)
        model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        scores.append(roc_auc_score(y[va_idx], model.predict_proba(X.iloc[va_idx])[:, 1]))
    return float(np.mean(scores))


def tune_lgb(X, y, n_trials):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return cv_auc_lgb(X, y, params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best LGB CV AUC: {study.best_value:.4f}")
    print(f"Best LGB params: {study.best_params}")
    return study.best_params


def cv_auc_cat(X, y, cat_feature_idx, params, n_splits=TUNE_CV_SPLITS):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in kf.split(X, y):
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_feature_idx)
        val_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_feature_idx)
        model = CatBoostClassifier(**params, task_type="GPU", devices="0",
                                    eval_metric="AUC", random_seed=RANDOM_STATE,
                                    early_stopping_rounds=50, verbose=False)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        scores.append(roc_auc_score(y[va_idx], model.predict_proba(val_pool)[:, 1]))
    return float(np.mean(scores))


def tune_cat(X, y, cat_feature_idx, n_trials):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "depth": trial.suggest_int("depth", 3, 8),  # GPU CatBoost caps at depth 8
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        return cv_auc_cat(X, y, cat_feature_idx, params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best CatBoost CV AUC: {study.best_value:.4f}")
    print(f"Best CatBoost params: {study.best_params}")
    return study.best_params


# --------------------------------------------------------------------------- #
# Final 5-fold OOF training
# --------------------------------------------------------------------------- #
def run_xgb_oof(X, y, X_test, best_params):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    importances = np.zeros(X.shape[1])
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                   random_state=RANDOM_STATE, early_stopping_rounds=50,
                                   **best_params, **XGB_GPU_BASE)
        model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        print(f"  [XGB] Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.4f}")
    print(f"[XGB] Overall OOF AUC: {roc_auc_score(y, oof):.4f}")
    return oof, test_preds, importances


def run_lgb_oof(X, y, X_test, best_params):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    importances = np.zeros(X.shape[1])
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        model = lgb.LGBMClassifier(objective="binary", metric="auc",
                                    random_state=RANDOM_STATE, verbosity=-1, **best_params)
        model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        print(f"  [LGB] Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.4f}")
    print(f"[LGB] Overall OOF AUC: {roc_auc_score(y, oof):.4f}")
    return oof, test_preds, importances


def run_cat_oof(X, y, X_test, best_params, cat_feature_idx):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    importances = np.zeros(X.shape[1])
    test_pool = Pool(X_test, cat_features=cat_feature_idx)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_feature_idx)
        val_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_feature_idx)
        model = CatBoostClassifier(**best_params, task_type="GPU", devices="0",
                                    eval_metric="AUC", random_seed=RANDOM_STATE,
                                    early_stopping_rounds=50, verbose=False)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        oof[va_idx] = model.predict_proba(val_pool)[:, 1]
        test_preds += model.predict_proba(test_pool)[:, 1] / N_SPLITS
        importances += np.array(model.get_feature_importance()) / N_SPLITS
        print(f"  [Cat] Fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.4f}")
    print(f"[Cat] Overall OOF AUC: {roc_auc_score(y, oof):.4f}")
    return oof, test_preds, importances


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")

    train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
    y = train[TARGET].values

    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    print("\nBuilding leak-free WOE / smoothed-target-encoded features...")
    woe_train, woe_test = build_woe_features(train_fe, test_fe, y)
    print(f"  Added {woe_train.shape[1]} WOE/composite features")

    cat_encoders = build_label_encoders(train_fe, RAW_CATEGORICAL + ENGINEERED_CATEGORICAL)
    train_enc = apply_label_encoders(train_fe, RAW_CATEGORICAL + ENGINEERED_CATEGORICAL, cat_encoders)
    test_enc = apply_label_encoders(test_fe, RAW_CATEGORICAL + ENGINEERED_CATEGORICAL, cat_encoders)

    base_numeric = RAW_NUMERIC + ENGINEERED_NUMERIC
    base_categorical = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL
    woe_cols = list(woe_train.columns)

    feature_cols = base_numeric + base_categorical + woe_cols

    X_xgb_lgb = pd.concat([train_enc[base_numeric + base_categorical], woe_train], axis=1)[feature_cols]
    X_test_xgb_lgb = pd.concat([test_enc[base_numeric + base_categorical], woe_test], axis=1)[feature_cols]

    X_cat = pd.concat([train_fe[base_numeric + base_categorical], woe_train], axis=1)[feature_cols]
    X_test_cat = pd.concat([test_fe[base_numeric + base_categorical], woe_test], axis=1)[feature_cols]
    for c in base_categorical:
        X_cat[c] = X_cat[c].astype(str)
        X_test_cat[c] = X_test_cat[c].astype(str)
    cat_feature_idx = [feature_cols.index(c) for c in base_categorical]

    print(f"\nTotal features: {len(feature_cols)}  "
          f"(base numeric: {len(base_numeric)}, base categorical: {len(base_categorical)}, WOE: {len(woe_cols)})")

    # ---------------- Robust tuning ---------------- #
    print("\n" + "=" * 80)
    print(f"TUNING XGBoost ({N_TRIALS_XGB} trials, {TUNE_CV_SPLITS}-fold CV objective)")
    print("=" * 80)
    xgb_params = tune_xgb(X_xgb_lgb, y, N_TRIALS_XGB)

    print("\n" + "=" * 80)
    print(f"TUNING LightGBM ({N_TRIALS_LGB} trials, {TUNE_CV_SPLITS}-fold CV objective)")
    print("=" * 80)
    lgb_params = tune_lgb(X_xgb_lgb, y, N_TRIALS_LGB)

    print("\n" + "=" * 80)
    print(f"TUNING CatBoost ({N_TRIALS_CAT} trials, {TUNE_CV_SPLITS}-fold CV objective)")
    print("=" * 80)
    cat_params = tune_cat(X_cat, y, cat_feature_idx, N_TRIALS_CAT)

    # ---------------- Final 5-fold OOF ---------------- #
    print("\n" + "=" * 80)
    print("XGBoost 5-fold OOF")
    print("=" * 80)
    oof_xgb, test_xgb, imp_xgb = run_xgb_oof(X_xgb_lgb, y, X_test_xgb_lgb, xgb_params)

    print("\n" + "=" * 80)
    print("LightGBM 5-fold OOF")
    print("=" * 80)
    oof_lgb, test_lgb, imp_lgb = run_lgb_oof(X_xgb_lgb, y, X_test_xgb_lgb, lgb_params)

    print("\n" + "=" * 80)
    print("CatBoost 5-fold OOF")
    print("=" * 80)
    oof_cat, test_cat, imp_cat = run_cat_oof(X_cat, y, X_test_cat, cat_params, cat_feature_idx)

    # ---------------- Feature importance ---------------- #
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (avg across folds, per model)")
    print("=" * 80)
    imp_df = pd.DataFrame({"feature": feature_cols, "xgb": imp_xgb, "lgb": imp_lgb, "cat": imp_cat})
    for col in ["xgb", "lgb", "cat"]:
        imp_df[col] = imp_df[col] / imp_df[col].sum()
    print(imp_df.sort_values("cat", ascending=False).to_string(index=False))

    # ---------------- Stack ---------------- #
    print("\n" + "=" * 80)
    print("STACKING")
    print("=" * 80)
    stack_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
    stack_test = np.column_stack([test_xgb, test_lgb, test_cat])

    meta = LogisticRegression(max_iter=1000)
    meta.fit(stack_train, y)
    stack_oof = meta.predict_proba(stack_train)[:, 1]
    stack_auc = roc_auc_score(y, stack_oof)
    print(f"Individual OOF AUCs -> XGB: {roc_auc_score(y, oof_xgb):.4f}  "
          f"LGB: {roc_auc_score(y, oof_lgb):.4f}  Cat: {roc_auc_score(y, oof_cat):.4f}")
    print(f"Meta-model weights: {dict(zip(['xgb', 'lgb', 'cat'], meta.coef_[0]))}")
    print(f"STACKED OOF AUC: {stack_auc:.4f}")

    test_final = meta.predict_proba(stack_test)[:, 1]
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved -> {SUBMISSION_PATH}")
    print(submission.head())

    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()