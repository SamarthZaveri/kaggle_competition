"""
Missingness-vs-Target + Feature-vs-Target Analysis
====================================================
Just set TRAIN_PATH below — full path, no splitting.
"""

import pandas as pd
import numpy as np
from scipy import stats

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv(TRAIN_PATH)

TARGET = "addicted_label"
ID_COL = "id"

feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

# ============================================================================ #
# PART 1: Is "missingness" itself informative? (MNAR check)
# ============================================================================ #
print("=" * 90)
print("PART 1: MISSINGNESS vs TARGET")
print("For each column, compare addicted_label rate when the value IS missing")
print("vs when it's present. Big gaps mean missingness is NOT random (MNAR) —")
print("worth keeping as a 'was_missing' flag feature rather than just filling it.")
print("=" * 90)

overall_rate = df[TARGET].mean()
print(f"\nOverall addicted_label rate: {overall_rate:.4f}\n")

rows = []
for col in feature_cols:
    is_missing = df[col].isna()
    n_missing = is_missing.sum()
    if n_missing == 0:
        continue

    rate_missing = df.loc[is_missing, TARGET].mean()
    rate_present = df.loc[~is_missing, TARGET].mean()
    diff = rate_missing - rate_present

    # chi-square test: is missingness independent of target?
    contingency = pd.crosstab(is_missing, df[TARGET])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)

    rows.append({
        "column": col,
        "n_missing": n_missing,
        "pct_missing": round(100 * n_missing / len(df), 2),
        "addicted_rate_when_missing": round(rate_missing, 4),
        "addicted_rate_when_present": round(rate_present, 4),
        "diff": round(diff, 4),
        "chi2_pvalue": p_value,
        "likely_MNAR": p_value < 0.01 and abs(diff) > 0.01,
    })

miss_report = pd.DataFrame(rows).sort_values("diff", key=abs, ascending=False)
print(miss_report.to_string(index=False))

print("\n--> Columns flagged likely_MNAR=True: worth adding a `<col>_was_missing`")
print("    binary feature even after imputing the value itself.")

# ============================================================================ #
# PART 2: Numeric features — distribution split by target
# ============================================================================ #
print("\n" + "=" * 90)
print("PART 2: NUMERIC FEATURES — addicted=1 vs addicted=0")
print("=" * 90)

rows = []
for col in numeric_cols:
    g0 = df.loc[df[TARGET] == 0, col].dropna()
    g1 = df.loc[df[TARGET] == 1, col].dropna()

    t_stat, p_value = stats.ttest_ind(g0, g1, equal_var=False)

    rows.append({
        "column": col,
        "mean_addicted_0": round(g0.mean(), 3),
        "mean_addicted_1": round(g1.mean(), 3),
        "diff": round(g1.mean() - g0.mean(), 3),
        "std_addicted_0": round(g0.std(), 3),
        "std_addicted_1": round(g1.std(), 3),
        "ttest_pvalue": p_value,
        "significant": p_value < 0.01,
    })

numeric_report = pd.DataFrame(rows).sort_values("diff", key=abs, ascending=False)
print(numeric_report.to_string(index=False))

# ============================================================================ #
# PART 3: Categorical features — proportion addicted by category
# ============================================================================ #
print("\n" + "=" * 90)
print("PART 3: CATEGORICAL FEATURES — addicted rate by category")
print("=" * 90)

for col in categorical_cols:
    print(f"\n--- {col} ---")
    ct = pd.crosstab(df[col], df[TARGET], normalize="index")
    ct.columns = ["rate_addicted_0", "rate_addicted_1"]
    counts = df[col].value_counts(dropna=False)
    ct["n"] = counts
    print(ct.sort_values("rate_addicted_1", ascending=False))

    # chi-square: is this category independent of target?
    contingency = pd.crosstab(df[col], df[TARGET])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    print(f"chi2 p-value: {p_value:.6f}  {'(significant)' if p_value < 0.01 else '(not significant)'}")

# ============================================================================ #
# PART 4: Correlation matrix among numeric features + target
# ============================================================================ #
print("\n" + "=" * 90)
print("PART 4: CORRELATION MATRIX (numeric features + target)")
print("=" * 90)
corr = df[numeric_cols + [TARGET]].corr()
print(corr[TARGET].sort_values(key=abs, ascending=False))