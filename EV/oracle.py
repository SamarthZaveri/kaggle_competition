"""
Oracle Ceiling Check
======================
Answers: "how high can AUC realistically go with this feature set?"
Uses leak-free (K-fold OOF) smoothed target encoding at increasing levels
of granularity -- no boosting, just group means -- so it runs in minutes
and tells us whether 0.97 is reachable before committing to another long
tuning run.

If AUC keeps climbing as we add more keys -> more feature/interaction
engineering will help.
If AUC plateaus early -> the ceiling is genuinely lower than 0.97 with
this feature set, and no amount of additional model tuning will close
the gap; that's a real, useful finding either way.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\train (2).csv"
TARGET = "Will_Buy_EV"
RANDOM_STATE = 42
N_SPLITS = 5
SMOOTHING = 50  # higher = more shrinkage toward global mean for small/rare groups


def oof_smoothed_target_encode(df, key_cols, y, n_splits=N_SPLITS, smoothing=SMOOTHING):
    """Leak-free OOF smoothed target encoding on a (possibly combined) key.
    Returns an array of OOF predicted probabilities, same length as df."""
    key = df[key_cols].astype(str).agg("_".join, axis=1) if len(key_cols) > 1 else df[key_cols[0]].astype(str)
    oof = np.zeros(len(df))
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for tr_idx, va_idx in kf.split(df, y):
        train_key = key.iloc[tr_idx]
        train_y = y[tr_idx]
        global_rate = train_y.mean()

        stats = pd.DataFrame({"key": train_key, "y": train_y}).groupby("key")["y"].agg(["sum", "count"])
        stats["smoothed_rate"] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)

        val_key = key.iloc[va_idx]
        oof[va_idx] = val_key.map(stats["smoothed_rate"]).fillna(global_rate).values

    return oof


def bin_numeric(s, q=5):
    return pd.qcut(s, q=q, duplicates="drop").astype(str)


def main():
    train = pd.read_csv(TRAIN_PATH)
    train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
    y = train[TARGET].values

    df = train.copy()
    df["income_bin"] = bin_numeric(df["Annual_Income_USD"], 5)
    df["age_bin"] = bin_numeric(df["Age"], 5)
    df["commute_bin"] = bin_numeric(df["Daily_Commute_km"], 5)
    df["chg_home_bin"] = bin_numeric(df["Charging_Stations_Near_Home"].astype(float), 5)
    df["chg_work_bin"] = bin_numeric(df["Charging_Stations_Near_Work"].astype(float), 5)

    print(f"Base rate: {y.mean():.4f}\n")

    key_sets = [
        ("Subsidy_Available", ["Subsidy_Available"]),
        ("Range_Anxiety_Level", ["Range_Anxiety_Level"]),
        ("Environmental_Concern_Level", ["Environmental_Concern_Level"]),
        ("Subsidy + RangeAnxiety", ["Subsidy_Available", "Range_Anxiety_Level"]),
        ("Subsidy + RangeAnxiety + EnvConcern", ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level"]),
        ("+ Home_Charging_Possible", ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level", "Home_Charging_Possible"]),
        ("+ income_bin", ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level", "Home_Charging_Possible", "income_bin"]),
        ("+ City_Type + Current_Car_Type", ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
                                             "Home_Charging_Possible", "income_bin", "City_Type", "Current_Car_Type"]),
        ("FULL CROSS (all raw + binned numeric)", ["Subsidy_Available", "Range_Anxiety_Level", "Environmental_Concern_Level",
                                                     "Home_Charging_Possible", "income_bin", "City_Type", "Current_Car_Type",
                                                     "Gender", "age_bin", "commute_bin", "chg_home_bin", "chg_work_bin"]),
    ]

    print("=" * 90)
    print(f"{'Key combination':<55} {'n_combos':>10} {'AUC':>10}")
    print("=" * 90)
    for name, cols in key_sets:
        oof_pred = oof_smoothed_target_encode(df, cols, y)
        auc = roc_auc_score(y, oof_pred)
        key = df[cols].astype(str).agg("_".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)
        n_combos = key.nunique()
        print(f"{name:<55} {n_combos:>10} {auc:>10.4f}")

    print("\nInterpretation:")
    print("  - If AUC is still climbing at 'FULL CROSS', richer interaction features will help.")
    print("  - If it plateaus well before FULL CROSS, that plateau is close to the real ceiling")
    print("    for this feature set -- further gains need either finer numeric detail or")
    print("    the raw model itself finding structure a coarse group-mean can't see.")


if __name__ == "__main__":
    main()