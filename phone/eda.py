"""
EDA Script — run this first, it will discover everything for you.
Just set TRAIN_PATH below to the FULL path of your file.
"""

import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
# EDIT THIS ONE LINE — full path to your file, nothing else needed
# --------------------------------------------------------------------------- #
TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv(TRAIN_PATH)

print("=" * 80)
print("SHAPE")
print("=" * 80)
print(df.shape)

print("\n" + "=" * 80)
print("COLUMN NAMES")
print("=" * 80)
print(list(df.columns))

print("\n" + "=" * 80)
print("DTYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("FIRST 5 ROWS")
print("=" * 80)
print(df.head())

print("\n" + "=" * 80)
print("MISSING VALUES PER COLUMN")
print("=" * 80)
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print(miss if not miss.empty else "None")

print("\n" + "=" * 80)
print("NUMERIC SUMMARY (.describe())")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("UNIQUE VALUE COUNTS PER COLUMN (to see which are categorical)")
print("=" * 80)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n" + "=" * 80)
print("VALUE COUNTS FOR LOW-CARDINALITY COLUMNS (<=20 unique)")
print("=" * 80)
for col in df.columns:
    if df[col].nunique() <= 20:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))