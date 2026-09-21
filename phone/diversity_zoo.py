"""
Diversity Zoo + KNN Lookup + Pseudo-Labeling Pipeline
=========================================================
Real architectural diversity: XGBoost, CatBoost (both tuned), ExtraTrees,
KNN classifier, Ridge/Logistic on polynomial features, plus a KNN-lookup
target feature and pseudo-labeling from confident test predictions.
"""

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostClassifier

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\test.csv"
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\submission_v3.csv"

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

PREDICTIVE_NUMERIC = ["daily_screen_time_hours", "weekend_screen_time",
                       "social_media_hours", "work_study_hours", "gaming_hours"]
NAIVE_NUMERIC = ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"]
NAIVE_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
STRONG_FEATS = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]


# --------------------------------------------------------------------------- #
# Imputation (unchanged)
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


# --------------------------------------------------------------------------- #
# Feature engineering (unchanged base + NEW KNN lookup feature)
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


def add_knn_lookup_feature(train_df, test_df, y_train, k=50, n_splits=5):
    """For each row, the mean target of its K nearest neighbors in strong-
    feature space. Leak-free: computed OOF for train via K-fold."""
    scaler = StandardScaler()
    X_train_strong = scaler.fit_transform(train_df[STRONG_FEATS])
    X_test_strong = scaler.transform(test_df[STRONG_FEATS])

    oof_knn_feat = np.zeros(len(train_df))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, va_idx in kf.split(X_train_strong, y_train):
        knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_strong[tr_idx], y_train[tr_idx])
        oof_knn_feat[va_idx] = knn.predict(X_train_strong[va_idx])

    knn_full = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
    knn_full.fit(X_train_strong, y_train)
    test_knn_feat = knn_full.predict(X_test_strong)

    return oof_knn_feat, test_knn_feat


# --------------------------------------------------------------------------- #
# Optuna tuning for CatBoost (new -- XGB already tuned)
# --------------------------------------------------------------------------- #
def tune_catboost(X, y, n_trials=25):
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=100_000, stratify=y, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 300, 1000),
            depth=trial.suggest_int("depth", 4, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            task_type="GPU", devices="0", verbose=False, random_seed=RANDOM_STATE,
        )
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, va_idx in kf.split(X_sub, y_sub):
            model = CatBoostClassifier(**params)
            model.fit(X_sub.iloc[tr_idx], y_sub.iloc[tr_idx])
            preds = model.predict_proba(X_sub.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_sub.iloc[va_idx], preds))
        return np.mean(scores)

    print(f"\nTuning CatBoost ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best CatBoost subsample AUC: {study.best_value:.4f}")
    print(f"Best CatBoost params: {study.best_params}")
    return study.best_params


# --------------------------------------------------------------------------- #
# Full diversity-zoo training with OOF stacking
# --------------------------------------------------------------------------- #
def train_diversity_zoo(X_train, y_train, X_test, best_xgb_params, best_cat_params):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # polynomial features for the Ridge/Logistic linear model (strong feats only, else explodes)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[STRONG_FEATS])
    X_test_poly = poly.transform(X_test[STRONG_FEATS])
    poly_scaler = StandardScaler()
    X_train_poly = poly_scaler.fit_transform(X_train_poly)
    X_test_poly = poly_scaler.transform(X_test_poly)

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y_train)
    oof = {k: np.zeros(n) for k in ["xgb", "cat", "extratrees", "knn", "ridge"]}
    test_preds = {k: [] for k in oof}

    xgb_params = {**best_xgb_params, **XGB_GPU_BASE, "random_state": RANDOM_STATE}
    cat_params = {**best_cat_params, "task_type": "GPU", "devices": "0",
                  "random_seed": RANDOM_STATE, "verbose": False}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
        # XGBoost
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xgb_params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["xgb"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["xgb"].append(m.predict_proba(X_test)[:, 1])

        # CatBoost
        m = CatBoostClassifier(**cat_params)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["cat"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["cat"].append(m.predict_proba(X_test)[:, 1])

        # ExtraTrees (different split randomness, CPU)
        m = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                  n_jobs=-1, random_state=RANDOM_STATE)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof["extratrees"][va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_preds["extratrees"].append(m.predict_proba(X_test)[:, 1])

        # KNN classifier (local structure, genuinely different inductive bias)
        m = KNeighborsClassifier(n_neighbors=75, n_jobs=-1)
        m.fit(X_train_scaled[tr_idx], y_train[tr_idx])
        oof["knn"][va_idx] = m.predict_proba(X_train_scaled[va_idx])[:, 1]
        test_preds["knn"].append(m.predict_proba(X_test_scaled)[:, 1])

        # Ridge-style logistic regression on polynomial strong features (pure additive/linear signal)
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(X_train_poly[tr_idx], y_train[tr_idx])
        oof["ridge"][va_idx] = m.predict_proba(X_train_poly[va_idx])[:, 1]
        test_preds["ridge"].append(m.predict_proba(X_test_poly)[:, 1])

        print(f"Fold {fold+1}: " + "  ".join(
            f"{k}={roc_auc_score(y_train[va_idx], oof[k][va_idx]):.4f}" for k in oof
        ))

    print("\n=== Overall OOF AUC per model ===")
    for k in oof:
        print(f"  {k}: {roc_auc_score(y_train, oof[k]):.4f}")

    test_pred_avg = {k: np.mean(test_preds[k], axis=0) for k in test_preds}
    return oof, test_pred_avg


def stack_and_predict(oof_dict, y_train, test_pred_dict):
    model_names = list(oof_dict.keys())
    stack_train = np.column_stack([oof_dict[k] for k in model_names])
    stack_test = np.column_stack([test_pred_dict[k] for k in model_names])

    meta = LogisticRegression(max_iter=1000)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y_train))
    for tr_idx, va_idx in kf.split(stack_train, y_train):
        meta_fold = LogisticRegression(max_iter=1000)
        meta_fold.fit(stack_train[tr_idx], y_train[tr_idx])
        stack_oof[va_idx] = meta_fold.predict_proba(stack_train[va_idx])[:, 1]

    stack_auc = roc_auc_score(y_train, stack_oof)
    print(f"\nStacked (diversity zoo) OOF AUC: {stack_auc:.4f}")

    meta.fit(stack_train, y_train)
    test_final = meta.predict_proba(stack_test)[:, 1]
    return test_final, stack_auc, stack_oof


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")

    train_imputed, test_imputed = run_imputation(train, test)
    train_final = engineer_features(train_imputed)
    test_final = engineer_features(test_imputed)

    y_train = train_final[TARGET].values

    print("\nAdding KNN lookup feature...")
    oof_knn_feat, test_knn_feat = add_knn_lookup_feature(train_final, test_final, y_train)
    train_final["knn_lookup_target"] = oof_knn_feat
    test_final["knn_lookup_target"] = test_knn_feat

    feature_cols = [c for c in train_final.columns if c not in (ID_COL, TARGET)]
    X_train = train_final[feature_cols].copy()
    X_test = test_final[feature_cols].copy()

    for c in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            le = LabelEncoder().fit(X_train[c].astype(str))
            X_train[c] = le.transform(X_train[c].astype(str))
            X_test[c] = X_test[c].astype(str).map(lambda v: le.transform([v])[0] if v in le.classes_ else -1)

    y_train_series = pd.Series(y_train)
    best_cat_params = tune_catboost(X_train, y_train_series, n_trials=25)

    print("\nTraining diversity zoo (XGB, CatBoost, ExtraTrees, KNN, Ridge-poly)...")
    oof, test_preds = train_diversity_zoo(X_train, y_train, X_test, BEST_XGB_PARAMS, best_cat_params)

    test_final_pred, stack_auc, stack_oof = stack_and_predict(oof, y_train, test_preds)

    # -------- Pseudo-labeling: add high-confidence test rows to training --------
    print("\nPseudo-labeling high-confidence test predictions...")
    confident_mask = (test_final_pred > 0.98) | (test_final_pred < 0.02)
    n_confident = confident_mask.sum()
    print(f"  {n_confident} / {len(test_final_pred)} test rows are high-confidence "
          f"({100*n_confident/len(test_final_pred):.1f}%)")

    if n_confident > 1000:
        pseudo_X = X_test[confident_mask].copy()
        pseudo_y = (test_final_pred[confident_mask] > 0.5).astype(int)

        X_train_aug = pd.concat([X_train, pseudo_X], ignore_index=True)
        y_train_aug = np.concatenate([y_train, pseudo_y])

        print(f"  Augmented training set: {len(X_train_aug)} rows (was {len(X_train)})")

        xgb_params = {**BEST_XGB_PARAMS, **XGB_GPU_BASE, "random_state": RANDOM_STATE}
        final_xgb = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **xgb_params)
        final_xgb.fit(X_train_aug, y_train_aug)
        pseudo_test_pred_xgb = final_xgb.predict_proba(X_test)[:, 1]

        cat_params = {**best_cat_params, "task_type": "GPU", "devices": "0",
                      "random_seed": RANDOM_STATE, "verbose": False}
        final_cat = CatBoostClassifier(**cat_params)
        final_cat.fit(X_train_aug, y_train_aug)
        pseudo_test_pred_cat = final_cat.predict_proba(X_test)[:, 1]

        # simple average of the two pseudo-labeled refits, blended with the stacked prediction
        pseudo_pred = (pseudo_test_pred_xgb + pseudo_test_pred_cat) / 2
        test_final_pred_v2 = 0.6 * test_final_pred + 0.4 * pseudo_pred
        print("  Pseudo-labeled refit blended in (60% original stack, 40% pseudo-label refit)")
    else:
        print("  Not enough high-confidence rows, skipping pseudo-labeling.")
        test_final_pred_v2 = test_final_pred

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_final_pred_v2})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved submission -> {SUBMISSION_PATH}")
    print(f"Final stacked OOF AUC (pre-pseudo-labeling): {stack_auc:.4f}")
    print(submission.head())


if __name__ == "__main__":
    main()