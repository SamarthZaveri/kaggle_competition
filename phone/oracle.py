import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

TRAIN_PATH = r"C:\Users\Samarth\Desktop\kaggle_phone_addiction\train (1).csv"
df = pd.read_csv(TRAIN_PATH)

strong = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]
df_clean = df.dropna(subset=strong + ["addicted_label"]).copy()
df_clean["composite"] = df_clean[strong].sum(axis=1)

# Fine-grained bins -> use each bin's empirical addicted-rate as the
# "oracle" probability for every row in that bin. This approximates the
# best AUC obtainable from composite alone, given the label noise.
df_clean["bin"] = pd.cut(df_clean["composite"], bins=100)
bin_rates = df_clean.groupby("bin", observed=True)["addicted_label"].transform("mean")

oracle_auc = roc_auc_score(df_clean["addicted_label"], bin_rates)
print(f"Oracle AUC from composite alone (upper bound estimate): {oracle_auc:.4f}")