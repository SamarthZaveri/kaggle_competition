"""
Diagnostic: is there a clean threshold rule hiding in the composite feature?
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
df = pd.read_csv(TRAIN_PATH)

strong = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]
df_clean = df.dropna(subset=strong + ["addicted_label"]).copy()
df_clean["composite"] = df_clean[strong].sum(axis=1)

y = df_clean["addicted_label"]

# 1. How well does composite ALONE separate the classes?
lr = LogisticRegression(max_iter=1000)
auc = cross_val_score(lr, df_clean[["composite"]], y, cv=5, scoring="roc_auc")
print(f"Composite alone (logistic reg): AUC = {auc.mean():.4f}")

# 2. Look for a sharp cutoff: addicted rate by composite decile/bin
df_clean["composite_bin"] = pd.cut(df_clean["composite"], bins=30)
rate_by_bin = df_clean.groupby("composite_bin", observed=True)["addicted_label"].agg(["mean", "count"])
print("\nAddicted rate by composite bin (look for a sharp jump from ~0 to ~1):")
print(rate_by_bin)

# 3. Try simple threshold rules directly — sweep candidate cutoffs
print("\nBest single-threshold rule on composite:")
best_acc, best_thresh = 0, None
for thresh in np.arange(df_clean["composite"].min(), df_clean["composite"].max(), 0.25):
    pred = (df_clean["composite"] > thresh).astype(int)
    acc = accuracy_score(y, pred)
    if acc > best_acc:
        best_acc, best_thresh = acc, thresh
print(f"Best threshold: {best_thresh:.2f}  ->  Accuracy = {best_acc:.4f}")

# 4. Check the individual columns' raw (non-imputed) rows only —
#    imputation noise could be softening an otherwise sharp rule
print(f"\nRows used (no imputation, complete cases only): {len(df_clean)} / {len(df)}")