"""
EDA: EV Purchase Prediction
=============================
Run this first and paste the full console output back — it drives every
decision in step 2 (imputation strategy, encoding, feature engineering,
which columns are worth tuning around).
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# --------------------------------------------------------------------------- #
# EDIT THESE
# --------------------------------------------------------------------------- #
TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\train (2).csv"
TEST_PATH = r"C:\Users\Samarth\Desktop\kaggle_competition\EV\test (1).csv"
TARGET = "Will_Buy_EV"
ID_COL = "id"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Target is "Yes"/"No" strings — map to 0/1 so groupby/corr means work.
TARGET_MAP = {"Yes": 1, "No": 0}
train[TARGET] = train[TARGET].map(TARGET_MAP)
assert train[TARGET].isna().sum() == 0, "Unexpected target values outside Yes/No"

print("=" * 80)
print("SHAPE")
print("=" * 80)
print(f"train: {train.shape}   test: {test.shape}")

print("\n" + "=" * 80)
print("DTYPES")
print("=" * 80)
print(train.dtypes)

print("\n" + "=" * 80)
print("HEAD")
print("=" * 80)
print(train.head(10))

# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("MISSING VALUES (train)")
print("=" * 80)
miss_train = train.isna().sum()
miss_train_pct = (miss_train / len(train) * 100).round(2)
miss_df = pd.DataFrame({"n_missing": miss_train, "pct_missing": miss_train_pct})
print(miss_df[miss_df.n_missing > 0].sort_values("n_missing", ascending=False))
if miss_df.n_missing.sum() == 0:
    print("No missing values in train.")

print("\n" + "=" * 80)
print("MISSING VALUES (test)")
print("=" * 80)
miss_test = test.isna().sum()
miss_test_pct = (miss_test / len(test) * 100).round(2)
miss_df_test = pd.DataFrame({"n_missing": miss_test, "pct_missing": miss_test_pct})
print(miss_df_test[miss_df_test.n_missing > 0].sort_values("n_missing", ascending=False))
if miss_df_test.n_missing.sum() == 0:
    print("No missing values in test.")

# --------------------------------------------------------------------------- #
# ID sanity checks
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("ID SANITY CHECKS")
print("=" * 80)
if ID_COL in train.columns:
    print(f"train id unique: {train[ID_COL].is_unique}")
    print(f"test id unique: {test[ID_COL].is_unique}")
    overlap = set(train[ID_COL]) & set(test[ID_COL])
    print(f"id overlap between train/test: {len(overlap)}")

# --------------------------------------------------------------------------- #
# Column consistency
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("COLUMN CONSISTENCY")
print("=" * 80)
train_cols = set(train.columns) - {TARGET}
test_cols = set(test.columns)
print(f"In train but not test: {train_cols - test_cols}")
print(f"In test but not train: {test_cols - train_cols}")

# --------------------------------------------------------------------------- #
# Target distribution
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print(f"TARGET DISTRIBUTION: {TARGET}")
print("=" * 80)
print(train[TARGET].value_counts())
print(train[TARGET].value_counts(normalize=True).round(4))

# --------------------------------------------------------------------------- #
# Split numeric vs categorical
# --------------------------------------------------------------------------- #
feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print("\n" + "=" * 80)
print("FEATURE TYPES")
print("=" * 80)
print(f"Numeric ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")

# --------------------------------------------------------------------------- #
# Numeric feature summary
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("NUMERIC SUMMARY STATS (train)")
print("=" * 80)
print(train[numeric_cols].describe().T)

# how "low-cardinality-numeric-that's-actually-categorical" check
print("\n" + "=" * 80)
print("NUMERIC COLUMN CARDINALITY (flag likely-categorical numerics)")
print("=" * 80)
for c in numeric_cols:
    nun = train[c].nunique()
    print(f"  {c}: {nun} unique values" + ("   <-- consider treating as categorical/ordinal" if nun <= 15 else ""))

# --------------------------------------------------------------------------- #
# Numeric vs target (mean target per quantile bin)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("NUMERIC FEATURES vs TARGET (mean target by quintile)")
print("=" * 80)
for c in numeric_cols:
    try:
        bins = pd.qcut(train[c], q=5, duplicates="drop")
        rate = train.groupby(bins, observed=True)[TARGET].agg(["mean", "count"])
        print(f"\n--- {c} ---")
        print(rate)
    except Exception as e:
        print(f"\n--- {c} --- (skipped: {e})")

# --------------------------------------------------------------------------- #
# Categorical cardinality + target rate per category
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("CATEGORICAL FEATURES: cardinality + target rate per category")
print("=" * 80)
for c in categorical_cols:
    nun = train[c].nunique()
    print(f"\n--- {c} ({nun} unique) ---")
    rate = train.groupby(c, observed=True)[TARGET].agg(["mean", "count"]).sort_values("count", ascending=False)
    print(rate.head(20))
    # check test has same categories
    test_only = set(test[c].dropna().unique()) - set(train[c].dropna().unique())
    if test_only:
        print(f"  Categories in test not seen in train: {test_only}")

# --------------------------------------------------------------------------- #
# Correlation matrix (numeric only, incl. target)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("CORRELATION WITH TARGET (numeric features, Pearson)")
print("=" * 80)
corr_with_target = train[numeric_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr_with_target)

print("\n" + "=" * 80)
print("FULL NUMERIC CORRELATION MATRIX")
print("=" * 80)
print(train[numeric_cols].corr().round(3))

# --------------------------------------------------------------------------- #
# Duplicate rows
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)
dupe_count = train.duplicated(subset=feature_cols).sum()
print(f"Duplicate feature-rows in train (ignoring id/target): {dupe_count}")

# --------------------------------------------------------------------------- #
# Quick interaction sniff: categorical x categorical target rate (top pairs)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 80)
print("PAIRWISE CATEGORICAL INTERACTIONS (target rate, top combos by count)")
print("=" * 80)
import itertools
for c1, c2 in itertools.combinations(categorical_cols, 2):
    combo = train.groupby([c1, c2], observed=True)[TARGET].agg(["mean", "count"]).sort_values("count", ascending=False)
    print(f"\n--- {c1} x {c2} ---")
    print(combo.head(8))

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)