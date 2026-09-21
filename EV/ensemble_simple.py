"""
EV Purchase Prediction: Simple Ensemble
======================================
Deliberately back to basics vs v2/v3: CatBoost + LightGBM + XGBoost +
Optuna + feature engineering, nothing more exotic (no WOE stack, no GLM).

The one real addition: v1/v2 only tried MULTIPLICATIVE interactions
(A * B). This adds FRACTION/RATIO interactions (A / (B+1)) for every
pair of numeric drivers that looked meaningful in the oracle_check /
forum analysis. Ratios carry different information than products --
e.g. income relative to commute cost, or charging access per car
owned -- and trees can't cheaply reconstruct a ratio from splits on
the raw columns the way they can reconstruct a threshold, so this is
a genuinely different signal source, not a repeat of what's plateaued.

Usage: fill in TRAIN_PATH / TEST_PATH, run it, paste the output back.
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
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\submission_ensemble_simple.csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
TUNE_CV_SPLITS = 3
N_TRIALS = 30

RAW_NUMERIC = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
               "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
               "Environmental_Concern_Level"]
RAW_CATEGORICAL = ["Gender", "City_Type", "Current_Car_Type",
                    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")


# --------------------------------------------------------------------------- #
# Feature engineering: multiply (kept from v1/v2) + fraction (new)
# --------------------------------------------------------------------------- #
def engineer_features(df):
    df = df.copy()

    # --- generator artifact flags (cheap, verified in the forum thread) ---
    df["is_30k_spike"] = (df["Annual_Income_USD"] == 30000.0).astype("int8")
    df["is_millionaire_cliff"] = (df["Annual_Income_USD"] >= 170537.0).astype("int8")
    df["is_dead_zone"] = ((df["Annual_Income_USD"] >= 31004.0) & (df["Annual_Income_USD"] <= 41970.0)).astype("int8")

    # --- charging access (kept from v1/v2) ---
    df["total_charging_access"] = df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    df["charging_home_work_ratio"] = df["Charging_Stations_Near_Home"] / (df["Charging_Stations_Near_Work"] + 1)

    # --- MULTIPLY interactions (kept from v1/v2 — already validated useful) ---
    df["income_x_env_concern"] = df["Annual_Income_USD"] * df["Environmental_Concern_Level"]
    df["commute_x_env_concern"] = df["Daily_Commute_km"] * df["Environmental_Concern_Level"]
    df["income_x_commute"] = df["Annual_Income_USD"] * df["Daily_Commute_km"]
    df["age_x_env_concern"] = df["Age"] * df["Environmental_Concern_Level"]
    df["cars_x_charging_home"] = df["Number_of_Cars_Owned"] * df["Charging_Stations_Near_Home"]

    # --- FRACTION / RATIO interactions (new — the thing we haven't tried) ---
    df["income_per_commute"] = df["Annual_Income_USD"] / (df["Daily_Commute_km"] + 1)
    df["income_per_age"] = df["Annual_Income_USD"] / (df["Age"] + 1)
    df["income_per_env_concern"] = df["Annual_Income_USD"] / (df["Environmental_Concern_Level"] + 1)
    df["income_per_car"] = df["Annual_Income_USD"] / (df["Number_of_Cars_Owned"] + 1)
    df["income_per_home_charging"] = df["Annual_Income_USD"] / (df["Charging_Stations_Near_Home"] + 1)
    df["income_per_work_charging"] = df["Annual_Income_USD"] / (df["Charging_Stations_Near_Work"] + 1)
    df["commute_per_age"] = df["Daily_Commute_km"] / (df["Age"] + 1)
    df["commute_per_car"] = df["Daily_Commute_km"] / (df["Number_of_Cars_Owned"] + 1)
    df["commute_per_home_charging"] = df["Daily_Commute_km"] / (df["Charging_Stations_Near_Home"] + 1)
    df["env_concern_per_car"] = df["Environmental_Concern_Level"] / (df["Number_of_Cars_Owned"] + 1)
    df["env_concern_per_commute"] = df["Environmental_Concern_Level"] / (df["Daily_Commute_km"] + 1)
    df["age_per_car"] = df["Age"] / (df["Number_of_Cars_Owned"] + 1)
    df["home_charging_per_car"] = df["Charging_Stations_Near_Home"] / (df["Number_of_Cars_Owned"] + 1)
    df["total_charging_per_car"] = df["total_charging_access"] / (df["Number_of_Cars_Owned"] + 1)
    df["total_charging_per_commute"] = df["total_charging_access"] / (df["Daily_Commute_km"] + 1)

    # --- categorical crosses (kept from v1/v2) ---
    df["subsidy_x_range_anxiety"] = df["Subsidy_Available"].astype(str) + "_" + df["Range_Anxiety_Level"].astype(str)
    df["subsidy_x_home_charging"] = df["Subsidy_Available"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)
    df["home_charging_x_range_anxiety"] = df["Home_Charging_Possible"].astype(str) + "_" + df["Range_Anxiety_Level"].astype(str)
    df["city_x_home_charging"] = df["City_Type"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)

    return df


ARTIFACT_FLAGS = ["is_30k_spike", "is_millionaire_cliff", "is_dead_zone"]
MULTIPLY_FEATURES = ["income_x_env_concern", "commute_x_env_concern", "income_x_commute",
                      "age_x_env_concern", "cars_x_charging_home"]
FRACTION_FEATURES = ["income_per_commute", "income_per_age", "income_per_env_concern",
                      "income_per_car", "income_per_home_charging", "income_per_work_charging",
                      "commute_per_age", "commute_per_car", "commute_per_home_charging",
                      "env_concern_per_car", "env_concern_per_commute", "age_per_car",
                      "home_charging_per_car", "total_charging_per_car", "total_charging_per_commute"]
CHARGING_FEATURES = ["total_charging_access", "charging_home_work_ratio"]
ENGINEERED_CATEGORICAL = ["subsidy_x_range_anxiety", "subsidy_x_home_charging",
                           "home_charging_x_range_anxiety", "city_x_home_charging"]

ENGINEERED_NUMERIC = ARTIFACT_FLAGS + CHARGING_FEATURES + MULTIPLY_FEATURES + FRACTION_FEATURES


# --------------------------------------------------------------------------- #
# Encoding
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
# Optuna tuning (3-fold CV objective)
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
            "max_depth": trial.suggest_int("max_depth", 3, 10),
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
            "depth": trial.suggest_int("depth", 3, 8),  # GPU CatBoost caps at 8
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
# Final 5-fold OOF
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

    base_numeric = RAW_NUMERIC + ENGINEERED_NUMERIC
    base_categorical = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL
    feature_cols = base_numeric + base_categorical

    print(f"\nTotal features: {len(feature_cols)}  "
          f"(numeric incl. {len(MULTIPLY_FEATURES)} multiply + {len(FRACTION_FEATURES)} fraction: "
          f"{len(base_numeric)}, categorical: {len(base_categorical)})")

    cat_encoders = build_label_encoders(train_fe, base_categorical)
    train_enc = apply_label_encoders(train_fe, base_categorical, cat_encoders)
    test_enc = apply_label_encoders(test_fe, base_categorical, cat_encoders)

    X_xgb_lgb = train_enc[feature_cols]
    X_test_xgb_lgb = test_enc[feature_cols]

    X_cat = train_fe[feature_cols].copy()
    X_test_cat = test_fe[feature_cols].copy()
    for c in base_categorical:
        X_cat[c] = X_cat[c].astype(str)
        X_test_cat[c] = X_test_cat[c].astype(str)
    cat_feature_idx = [feature_cols.index(c) for c in base_categorical]

    # ---------------- Tuning ---------------- #
    print("\n" + "=" * 80)
    print(f"TUNING XGBoost ({N_TRIALS} trials, {TUNE_CV_SPLITS}-fold CV objective)")
    print("=" * 80)
    xgb_params = tune_xgb(X_xgb_lgb, y, N_TRIALS)

    print("\n" + "=" * 80)
    print(f"TUNING LightGBM ({N_TRIALS} trials, {TUNE_CV_SPLITS}-fold CV objective)")
    print("=" * 80)
    lgb_params = tune_lgb(X_xgb_lgb, y, N_TRIALS)

    print("\n" + "=" * 80)
    print(f"TUNING CatBoost ({N_TRIALS} trials, {TUNE_CV_SPLITS}-fold CV objective)")
    print("=" * 80)
    cat_params = tune_cat(X_cat, y, cat_feature_idx, N_TRIALS)

    # ---------------- Final 5-fold OOF ---------------- #
    print("\n" + "=" * 80 + "\nXGBoost 5-fold OOF\n" + "=" * 80)
    oof_xgb, test_xgb, imp_xgb = run_xgb_oof(X_xgb_lgb, y, X_test_xgb_lgb, xgb_params)

    print("\n" + "=" * 80 + "\nLightGBM 5-fold OOF\n" + "=" * 80)
    oof_lgb, test_lgb, imp_lgb = run_lgb_oof(X_xgb_lgb, y, X_test_xgb_lgb, lgb_params)

    print("\n" + "=" * 80 + "\nCatBoost 5-fold OOF\n" + "=" * 80)
    oof_cat, test_cat, imp_cat = run_cat_oof(X_cat, y, X_test_cat, cat_params, cat_feature_idx)

    # ---------------- Feature importance ---------------- #
    print("\n" + "=" * 80 + "\nFEATURE IMPORTANCE (avg across folds, per model)\n" + "=" * 80)
    imp_df = pd.DataFrame({"feature": feature_cols, "xgb": imp_xgb, "lgb": imp_lgb, "cat": imp_cat})
    for col in ["xgb", "lgb", "cat"]:
        imp_df[col] = imp_df[col] / imp_df[col].sum()
    imp_df["is_fraction_feature"] = imp_df["feature"].isin(FRACTION_FEATURES)
    print(imp_df.sort_values("cat", ascending=False).to_string(index=False))
    frac_share = imp_df.loc[imp_df["is_fraction_feature"], ["xgb", "lgb", "cat"]].sum()
    print(f"\nShare of total importance held by FRACTION features -> "
          f"xgb: {frac_share['xgb']:.3f}  lgb: {frac_share['lgb']:.3f}  cat: {frac_share['cat']:.3f}")

    # ---------------- Stack ---------------- #
    print("\n" + "=" * 80 + "\nSTACKING\n" + "=" * 80)
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

    # --- known deterministic-region post-processing (verified in forum thread) ---
    test_final = meta.predict_proba(stack_test)[:, 1]
    test_final = np.where(test_fe["is_millionaire_cliff"] == 1, 1.0, test_final)
    test_final = np.where(test_fe["is_dead_zone"] == 1, 0.0, test_final)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved -> {SUBMISSION_PATH}")
    print(submission.head())
    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()