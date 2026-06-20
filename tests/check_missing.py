# tests/check_missing.py

import pandas as pd

df = pd.read_csv("data/raw/job_dataset.csv")

print(df["Title"].isna().sum())
print(df[df["Title"].isna()])