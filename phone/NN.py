"""
PyTorch NN with Categorical Embeddings — GPU (RTX 5050, cu128)
==================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\test.csv"
SUBMISSION_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\submission_nn.csv"

TARGET = "addicted_label"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

XGB_GPU_BASE = dict(tree_method="hist", device="cuda")

PREDICTIVE_NUMERIC = ["daily_screen_time_hours", "weekend_screen_time",
                       "social_media_hours", "work_study_hours", "gaming_hours"]
NAIVE_NUMERIC = ["age", "sleep_hours", "notifications_per_day", "app_opens_per_day"]
NAIVE_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]

CAT_COLS_FOR_EMBEDDING = ["gender", "stress_level", "academic_work_impact",
                           "age_bucket", "gender_x_stress", "academic_x_work"]
NUMERIC_COLS_FOR_NN = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time", "weekend_vs_daily_diff", "social_media_share",
    "total_signal_composite", "screen_to_work_ratio", "composite_x_social_share",
    "composite_squared", "composite_logit", "age_bucket_x_composite",
    "notif_x_appopen", "notif_per_appopen", "sleep_hours_sq", "notifications_sq",
]

import xgboost as xgb


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

    df["age_bucket"] = pd.cut(df["age"], bins=[17, 21, 25, 29, 33, 36], labels=[0, 1, 2, 3, 4]).astype(str)
    df["age_bucket_x_composite"] = df["age_bucket"].astype(float) * df["total_signal_composite"]

    df["notif_x_appopen"] = df["notifications_per_day"] * df["app_opens_per_day"]
    df["notif_per_appopen"] = (df["notifications_per_day"] /
                                df["app_opens_per_day"].replace(0, np.nan)).fillna(0)

    df["gender_x_stress"] = df["gender"].astype(str) + "_" + df["stress_level"].astype(str)
    df["academic_x_work"] = (df["academic_work_impact"].astype(str) + "_" +
                              pd.cut(df["work_study_hours"], bins=4).astype(str))

    df["sleep_hours_sq"] = df["sleep_hours"] ** 2
    df["notifications_sq"] = df["notifications_per_day"] ** 2
    return df


class EmbeddingNN(nn.Module):
    def __init__(self, cat_cardinalities, n_numeric):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(50, (card + 1) // 2)) for card in cat_cardinalities
        ])
        emb_dim_total = sum(min(50, (card + 1) // 2) for card in cat_cardinalities)
        input_dim = emb_dim_total + n_numeric

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x).squeeze(-1)


def train_nn_fold(X_cat_tr, X_num_tr, y_tr, X_cat_va, X_num_va, y_va,
                   cat_cardinalities, epochs=40, batch_size=4096):
    model = EmbeddingNN(cat_cardinalities, X_num_tr.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.tensor(X_cat_tr, dtype=torch.long),
        torch.tensor(X_num_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    X_cat_va_t = torch.tensor(X_cat_va, dtype=torch.long).to(DEVICE)
    X_num_va_t = torch.tensor(X_num_va, dtype=torch.float32).to(DEVICE)

    best_auc, best_state, patience, bad_epochs = 0, None, 8, 0
    for epoch in range(epochs):
        model.train()
        for xb_cat, xb_num, yb in train_loader:
            xb_cat, xb_num, yb = xb_cat.to(DEVICE), xb_num.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb_cat, xb_num), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_cat_va_t, X_num_va_t).cpu().numpy()
        val_proba = 1 / (1 + np.exp(-val_logits))
        val_auc = roc_auc_score(y_va, val_proba)

        if val_auc > best_auc:
            best_auc, best_state, bad_epochs = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if epoch % 5 == 0:
            print(f"    epoch {epoch}: val_auc={val_auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(X_cat_va_t, X_num_va_t).cpu().numpy()
    return 1 / (1 + np.exp(-final_logits)), best_auc, model


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train: {train.shape}   test: {test.shape}")

    train_imputed, test_imputed = run_imputation(train, test)
    train_final = engineer_features(train_imputed)
    test_final = engineer_features(test_imputed)

    y_train = train_final[TARGET].values.astype(np.float32)

    # encode categoricals for embeddings (need contiguous 0..N-1 integer ids)
    cat_encoders = {}
    cat_cardinalities = []
    combined_cat = pd.concat([train_final[CAT_COLS_FOR_EMBEDDING], test_final[CAT_COLS_FOR_EMBEDDING]])
    for c in CAT_COLS_FOR_EMBEDDING:
        le = LabelEncoder().fit(combined_cat[c].astype(str))
        cat_encoders[c] = le
        cat_cardinalities.append(len(le.classes_))

    X_cat_train = np.column_stack([
        cat_encoders[c].transform(train_final[c].astype(str)) for c in CAT_COLS_FOR_EMBEDDING
    ])
    X_cat_test = np.column_stack([
        cat_encoders[c].transform(test_final[c].astype(str)) for c in CAT_COLS_FOR_EMBEDDING
    ])

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train_final[NUMERIC_COLS_FOR_NN]).astype(np.float32)
    X_num_test = scaler.transform(test_final[NUMERIC_COLS_FOR_NN]).astype(np.float32)

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_nn = np.zeros(len(y_train))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cat_train, y_train)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        val_pred, val_auc, model = train_nn_fold(
            X_cat_train[tr_idx], X_num_train[tr_idx], y_train[tr_idx],
            X_cat_train[va_idx], X_num_train[va_idx], y_train[va_idx],
            cat_cardinalities,
        )
        oof_nn[va_idx] = val_pred
        print(f"  Fold {fold+1} best AUC: {val_auc:.4f}")

        model.eval()
        with torch.no_grad():
            test_logits = model(
                torch.tensor(X_cat_test, dtype=torch.long).to(DEVICE),
                torch.tensor(X_num_test, dtype=torch.float32).to(DEVICE),
            ).cpu().numpy()
        test_preds.append(1 / (1 + np.exp(-test_logits)))

    print(f"\nOverall NN OOF AUC: {roc_auc_score(y_train, oof_nn):.4f}")

    test_pred_final = np.mean(test_preds, axis=0)
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved -> {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()