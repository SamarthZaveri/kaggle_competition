"""
EV Purchase Prediction: Ensemble v4 -- Safe Fold-Nested Dual TE
===================================================================
Fixes the v3 leak (AUC 1.0000 OOF / 0.51 LB) at the root:

v3's bug: LOO encoding was computed ONCE on the full dataset (leaving out
only the individual row). For any downstream CV fold, that means the
"training" features for that fold were built using data that includes
the fold's OWN held-out validation rows -- a classic naive-target-encoding
leak, just disguised as "leave-one-out". Rich, high-cardinality combo
keys made this catastrophic; even the low-cardinality single keys leaked
a little (that's a real bug, not just imprecise smoothing).

v4's fix: every encoding -- both the K-fold-style value and the LOO value
-- is rebuilt FRESH per fold, using ONLY that fold's training partition.
Validation-fold rows never contribute to any encoding used to predict
them, at any point in the pipeline, for any key. This is architecturally
guaranteed rather than incidentally true.

Also: tuning and final training now share the SAME 5-fold split (v1-v3
used the final 5-fold split for training but an independent 3-fold split
for tuning -- that mismatch was a milder version of the same leak
pathway that broke LOO, just diluted by K-fold's smoothing). Precomputing
the fold-safe features once (they don't depend on hyperparameters) means
this doesn't cost extra runtime despite being fit fresh per fold.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\submission_ensemble_v4_safe.csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5          # single fold split used for BOTH tuning and final training
N_TRIALS_XGB = 25
N_TRIALS_LGB = 25
N_TRIALS_CAT = 25
TE_SMOOTHING = 50

RAW_NUMERIC = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
               "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
               "Environmental_Concern_Level"]
RAW_CATEGORICAL = ["Gender", "City_Type", "Current_Car_Type",
                    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")


# --------------------------------------------------------------------------- #
# Feature engineering (unchanged)
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


def fit_bin_edges(series, q=5):
    _, edges = pd.qcut(series, q=q, duplicates="drop", retbins=True)
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bins(series, edges):
    return pd.cut(series, bins=edges, labels=False).astype(str)


TE_SINGLE_KEYS = ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
                   "Home_Charging_Possible", "City_Type", "Current_Car_Type", "Gender",
                   "income_bin", "income_bin_fine", "age_bin", "commute_bin"]
TE_COMBO_KEYS = [
    ["Subsidy_Available", "Range_Anxiety_Level"],
    ["Subsidy_Available", "Home_Charging_Possible"],
    ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level", "Home_Charging_Possible"],
    ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level", "Home_Charging_Possible", "income_bin"],
]
MASTER_SCORE_KEYS = ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
                      "Home_Charging_Possible", "income_bin"]


def to_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def smoothed_rate_stats(key, y, smoothing):
    global_rate = y.mean()
    stats = pd.DataFrame({"key": key.values, "y": y}).groupby("key")["y"].agg(["sum", "count"])
    stats["rate"] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    return stats["rate"], global_rate


def loo_within(key, y, smoothing):
    df = pd.DataFrame({"key": key.values, "y": y})
    global_rate = y.mean()
    sum_map = df.groupby("key")["y"].transform("sum")
    count_map = df.groupby("key")["y"].transform("count")
    loo = (sum_map - df["y"] + smoothing * global_rate) / (count_map - 1 + smoothing)
    return loo.values.astype(float)


def build_fold_safe_features(train_df, test_df, y, tr_idx, va_idx, keys_single, keys_combo, smoothing=TE_SMOOTHING):
    tr_out = pd.DataFrame(index=train_df.index[tr_idx])
    va_out = pd.DataFrame(index=train_df.index[va_idx])
    test_out = pd.DataFrame(index=test_df.index)

    y_tr = y[tr_idx]

    def get_key_series(df, idx_slice, cols):
        if len(cols) == 1:
            base = df[cols[0]].astype(str)
        else:
            base = df[cols].astype(str).agg("_".join, axis=1)
        return base.iloc[idx_slice] if idx_slice is not None else base

    master_kfold_tr = np.zeros(len(tr_idx))
    master_kfold_va = np.zeros(len(va_idx))
    master_kfold_test = np.zeros(len(test_df))
    master_loo_tr = np.zeros(len(tr_idx))

    def process(name, cols):
        tr_key = get_key_series(train_df, tr_idx, cols)
        va_key = get_key_series(train_df, va_idx, cols)
        test_key = get_key_series(test_df, None, cols)

        rate_lookup, global_rate = smoothed_rate_stats(tr_key, y_tr, smoothing)
        kfold_tr = tr_key.map(rate_lookup).fillna(global_rate).values
        kfold_va = va_key.map(rate_lookup).fillna(global_rate).values
        kfold_test = test_key.map(rate_lookup).fillna(global_rate).values

        loo_tr = loo_within(tr_key, y_tr, smoothing)

        tr_out[f"te_kfold_{name}"] = kfold_tr
        tr_out[f"te_loo_{name}"] = loo_tr
        va_out[f"te_kfold_{name}"] = kfold_va
        va_out[f"te_loo_{name}"] = kfold_va
        test_out[f"te_kfold_{name}"] = kfold_test
        test_out[f"te_loo_{name}"] = kfold_test

        return kfold_tr, kfold_va, kfold_test, loo_tr

    income_edges = fit_bin_edges(train_df["Annual_Income_USD"].iloc[tr_idx], 5)
    income_edges_fine = fit_bin_edges(train_df["Annual_Income_USD"].iloc[tr_idx], 10)
    age_edges = fit_bin_edges(train_df["Age"].iloc[tr_idx], 5)
    commute_edges = fit_bin_edges(train_df["Daily_Commute_km"].iloc[tr_idx], 5)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["income_bin"] = apply_bins(train_df["Annual_Income_USD"], income_edges)
    train_df["income_bin_fine"] = apply_bins(train_df["Annual_Income_USD"], income_edges_fine)
    train_df["age_bin"] = apply_bins(train_df["Age"], age_edges)
    train_df["commute_bin"] = apply_bins(train_df["Daily_Commute_km"], commute_edges)
    test_df["income_bin"] = apply_bins(test_df["Annual_Income_USD"], income_edges)
    test_df["income_bin_fine"] = apply_bins(test_df["Annual_Income_USD"], income_edges_fine)
    test_df["age_bin"] = apply_bins(test_df["Age"], age_edges)
    test_df["commute_bin"] = apply_bins(test_df["Daily_Commute_km"], commute_edges)

    for k in keys_single:
        kfold_tr, kfold_va, kfold_test, loo_tr = process(k, [k])
        if k in MASTER_SCORE_KEYS:
            master_kfold_tr += to_logit(kfold_tr)
            master_kfold_va += to_logit(kfold_va)
            master_kfold_test += to_logit(kfold_test)
            master_loo_tr += to_logit(loo_tr)

    for combo in keys_combo:
        name = "_x_".join(combo)
        process(name, combo)

    tr_out["master_logit_kfold"] = master_kfold_tr
    va_out["master_logit_kfold"] = master_kfold_va
    test_out["master_logit_kfold"] = master_kfold_test
    tr_out["master_logit_loo"] = master_loo_tr
    va_out["master_logit_loo"] = master_kfold_va
    test_out["master_logit_loo"] = master_kfold_test

    return tr_out, va_out, test_out


def build_label_encoders(train_df, cat_cols):
    return {c: LabelEncoder().fit(train_df[c].astype(str)) for c in cat_cols}


def apply_label_encoders(df, cat_cols, encoders):
    out = df.copy()
    for c in cat_cols:
        le = encoders[c]
        out[c] = out[c].astype(str).map(lambda v: (le.transform([v])[0] if v in le.classes_ else -1))
    return out


def precompute_all_folds(train_fe, test_fe, y, base_numeric, base_categorical, cat_encoders):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_data = []

    train_le = apply_label_encoders(train_fe, base_categorical, cat_encoders)
    test_le = apply_label_encoders(test_fe, base_categorical, cat_encoders)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_fe, y)):
        print(f"  Building safe features for fold {fold+1}/{N_SPLITS}...")
        tr_te, va_te, test_te = build_fold_safe_features(
            train_fe, test_fe, y, tr_idx, va_idx, TE_SINGLE_KEYS, TE_COMBO_KEYS
        )
        te_cols = list(tr_te.columns)
        feature_cols = base_numeric + base_categorical + te_cols

        X_tr_xl = pd.concat([train_le.iloc[tr_idx][base_numeric + base_categorical].reset_index(drop=True),
                              tr_te.reset_index(drop=True)], axis=1)[feature_cols]
        X_va_xl = pd.concat([train_le.iloc[va_idx][base_numeric + base_categorical].reset_index(drop=True),
                              va_te.reset_index(drop=True)], axis=1)[feature_cols]
        X_test_xl = pd.concat([test_le[base_numeric + base_categorical].reset_index(drop=True),
                                test_te.reset_index(drop=True)], axis=1)[feature_cols]

        X_tr_cat = pd.concat([train_fe.iloc[tr_idx][base_numeric + base_categorical].reset_index(drop=True),
                               tr_te.reset_index(drop=True)], axis=1)[feature_cols]
        X_va_cat = pd.concat([train_fe.iloc[va_idx][base_numeric + base_categorical].reset_index(drop=True),
                               va_te.reset_index(drop=True)], axis=1)[feature_cols]
        X_test_cat = pd.concat([test_fe[base_numeric + base_categorical].reset_index(drop=True),
                                 test_te.reset_index(drop=True)], axis=1)[feature_cols]
        for c in base_categorical:
            X_tr_cat[c] = X_tr_cat[c].astype(str)
            X_va_cat[c] = X_va_cat[c].astype(str)
            X_test_cat[c] = X_test_cat[c].astype(str)
        cat_feature_idx = [feature_cols.index(c) for c in base_categorical]

        fold_data.append(dict(
            tr_idx=tr_idx, va_idx=va_idx, feature_cols=feature_cols,
            X_tr_xl=X_tr_xl, X_va_xl=X_va_xl, X_test_xl=X_test_xl,
            X_tr_cat=X_tr_cat, X_va_cat=X_va_cat, X_test_cat=X_test_cat,
            cat_feature_idx=cat_feature_idx, y_tr=y[tr_idx], y_va=y[va_idx],
        ))

    return fold_data


def tune_xgb(fold_data, n_trials):
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
        scores = []
        for fd in fold_data:
            model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                       random_state=RANDOM_STATE, early_stopping_rounds=50,
                                       **params, **XGB_GPU_BASE)
            model.fit(fd["X_tr_xl"], fd["y_tr"], eval_set=[(fd["X_va_xl"], fd["y_va"])], verbose=False)
            scores.append(roc_auc_score(fd["y_va"], model.predict_proba(fd["X_va_xl"])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best XGB CV AUC: {study.best_value:.4f}")
    print(f"Best XGB params: {study.best_params}")
    return study.best_params


def tune_lgb(fold_data, n_trials):
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
        scores = []
        for fd in fold_data:
            model = lgb.LGBMClassifier(objective="binary", metric="auc",
                                        random_state=RANDOM_STATE, verbosity=-1, **params)
            model.fit(fd["X_tr_xl"], fd["y_tr"], eval_set=[(fd["X_va_xl"], fd["y_va"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            scores.append(roc_auc_score(fd["y_va"], model.predict_proba(fd["X_va_xl"])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best LGB CV AUC: {study.best_value:.4f}")
    print(f"Best LGB params: {study.best_params}")
    return study.best_params


def tune_cat(fold_data, n_trials):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
        scores = []
        for fd in fold_data:
            train_pool = Pool(fd["X_tr_cat"], fd["y_tr"], cat_features=fd["cat_feature_idx"])
            val_pool = Pool(fd["X_va_cat"], fd["y_va"], cat_features=fd["cat_feature_idx"])
            model = CatBoostClassifier(**params, task_type="GPU", devices="0",
                                        eval_metric="AUC", random_seed=RANDOM_STATE,
                                        early_stopping_rounds=50, verbose=False)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            scores.append(roc_auc_score(fd["y_va"], model.predict_proba(val_pool)[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best CatBoost CV AUC: {study.best_value:.4f}")
    print(f"Best CatBoost params: {study.best_params}")
    return study.best_params


def run_xgb_final(fold_data, best_params, y_full):
    oof = np.zeros(len(y_full))
    test_preds = np.zeros(len(fold_data[0]["X_test_xl"]))
    importances = np.zeros(len(fold_data[0]["feature_cols"]))
    for i, fd in enumerate(fold_data):
        model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                   random_state=RANDOM_STATE, early_stopping_rounds=50,
                                   **best_params, **XGB_GPU_BASE)
        model.fit(fd["X_tr_xl"], fd["y_tr"], eval_set=[(fd["X_va_xl"], fd["y_va"])], verbose=False)
        preds = model.predict_proba(fd["X_va_xl"])[:, 1]
        oof[fd["va_idx"]] = preds
        test_preds += model.predict_proba(fd["X_test_xl"])[:, 1] / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        print(f"  [XGB] Fold {i+1} AUC: {roc_auc_score(fd['y_va'], preds):.4f}")
    print(f"[XGB] Overall OOF AUC: {roc_auc_score(y_full, oof):.4f}")
    return oof, test_preds, importances


def run_lgb_final(fold_data, best_params, y_full):
    oof = np.zeros(len(y_full))
    test_preds = np.zeros(len(fold_data[0]["X_test_xl"]))
    importances = np.zeros(len(fold_data[0]["feature_cols"]))
    for i, fd in enumerate(fold_data):
        model = lgb.LGBMClassifier(objective="binary", metric="auc",
                                    random_state=RANDOM_STATE, verbosity=-1, **best_params)
        model.fit(fd["X_tr_xl"], fd["y_tr"], eval_set=[(fd["X_va_xl"], fd["y_va"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict_proba(fd["X_va_xl"])[:, 1]
        oof[fd["va_idx"]] = preds
        test_preds += model.predict_proba(fd["X_test_xl"])[:, 1] / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        print(f"  [LGB] Fold {i+1} AUC: {roc_auc_score(fd['y_va'], preds):.4f}")
    print(f"[LGB] Overall OOF AUC: {roc_auc_score(y_full, oof):.4f}")
    return oof, test_preds, importances


def run_cat_final(fold_data, best_params, y_full):
    oof = np.zeros(len(y_full))
    test_preds = np.zeros(len(fold_data[0]["X_test_cat"]))
    importances = np.zeros(len(fold_data[0]["feature_cols"]))
    for i, fd in enumerate(fold_data):
        train_pool = Pool(fd["X_tr_cat"], fd["y_tr"], cat_features=fd["cat_feature_idx"])
        val_pool = Pool(fd["X_va_cat"], fd["y_va"], cat_features=fd["cat_feature_idx"])
        test_pool = Pool(fd["X_test_cat"], cat_features=fd["cat_feature_idx"])
        model = CatBoostClassifier(**best_params, task_type="GPU", devices="0",
                                    eval_metric="AUC", random_seed=RANDOM_STATE,
                                    early_stopping_rounds=50, verbose=False)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        preds = model.predict_proba(val_pool)[:, 1]
        oof[fd["va_idx"]] = preds
        test_preds += model.predict_proba(test_pool)[:, 1] / N_SPLITS
        importances += np.array(model.get_feature_importance()) / N_SPLITS
        print(f"  [Cat] Fold {i+1} AUC: {roc_auc_score(fd['y_va'], preds):.4f}")
    print(f"[Cat] Overall OOF AUC: {roc_auc_score(y_full, oof):.4f}")
    return oof, test_preds, importances


def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")

    train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
    y = train[TARGET].values

    train_fe = engineer_features(train).reset_index(drop=True)
    test_fe = engineer_features(test).reset_index(drop=True)

    base_numeric = RAW_NUMERIC + ENGINEERED_NUMERIC
    base_categorical = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL
    cat_encoders = build_label_encoders(train_fe, base_categorical)

    print("\nPrecomputing fold-safe dual TE features (once, reused for tuning + final)...")
    fold_data = precompute_all_folds(train_fe, test_fe, y, base_numeric, base_categorical, cat_encoders)
    n_features = len(fold_data[0]["feature_cols"])
    print(f"Total features per fold: {n_features}")

    print("\n" + "=" * 80)
    print(f"TUNING XGBoost ({N_TRIALS_XGB} trials, {N_SPLITS}-fold CV objective)")
    print("=" * 80)
    xgb_params = tune_xgb(fold_data, N_TRIALS_XGB)

    print("\n" + "=" * 80)
    print(f"TUNING LightGBM ({N_TRIALS_LGB} trials, {N_SPLITS}-fold CV objective)")
    print("=" * 80)
    lgb_params = tune_lgb(fold_data, N_TRIALS_LGB)

    print("\n" + "=" * 80)
    print(f"TUNING CatBoost ({N_TRIALS_CAT} trials, {N_SPLITS}-fold CV objective)")
    print("=" * 80)
    cat_params = tune_cat(fold_data, N_TRIALS_CAT)

    print("\n" + "=" * 80)
    print("XGBoost final")
    print("=" * 80)
    oof_xgb, test_xgb, imp_xgb = run_xgb_final(fold_data, xgb_params, y)

    print("\n" + "=" * 80)
    print("LightGBM final")
    print("=" * 80)
    oof_lgb, test_lgb, imp_lgb = run_lgb_final(fold_data, lgb_params, y)

    print("\n" + "=" * 80)
    print("CatBoost final")
    print("=" * 80)
    oof_cat, test_cat, imp_cat = run_cat_final(fold_data, cat_params, y)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (avg across folds) -- top 25")
    print("=" * 80)
    feature_cols = fold_data[0]["feature_cols"]
    imp_df = pd.DataFrame({"feature": feature_cols, "xgb": imp_xgb, "lgb": imp_lgb, "cat": imp_cat})
    for col in ["xgb", "lgb", "cat"]:
        imp_df[col] = imp_df[col] / imp_df[col].sum()
    print(imp_df.sort_values("cat", ascending=False).head(25).to_string(index=False))

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

    if stack_auc > 0.97 or any(roc_auc_score(y, o) > 0.97 for o in [oof_xgb, oof_lgb, oof_cat]):
        print("\n*** WARNING: AUC above 0.97 is far outside the community's reported range")
        print("*** (~0.94-0.947). Treat this as a probable leak, not a breakthrough, and")
        print("*** do not submit before investigating further. ***")

    test_final = meta.predict_proba(stack_test)[:, 1]
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved -> {SUBMISSION_PATH}")
    print(submission.head())

    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()