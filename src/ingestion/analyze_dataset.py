import pandas as pd

df = pd.read_csv(
    "data/raw/job_dataset.csv"
)

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATES")
print(df.duplicated().sum())

print("\nEXPERIENCE LEVEL DISTRIBUTION")
print(df["ExperienceLevel"].value_counts())

print("\nTOP 20 JOB TITLES")
print(df["Title"].value_counts().head(20))