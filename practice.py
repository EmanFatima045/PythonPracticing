import pandas as pd
import numpy as np

df = pd.read_csv("student-attendance.csv", on_bad_lines='skip')

print("Original Data:\n", df.head())
df = df[df["Hours_Studied"] != "Hours_Studied"]

df = df[[
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores"
]]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.fillna(df.mean())
df = df.drop_duplicates()

def remove_outliers(dataframe, columns, threshold=3):
    for col in columns:
        mean = dataframe[col].mean()
        std = dataframe[col].std()
        z_scores = (dataframe[col] - mean) / std
        dataframe = dataframe[np.abs(z_scores) < threshold]
    return dataframe

df = remove_outliers(df, df.columns)
df = df.reset_index(drop=True)

df = df.sort_values(by="Attendance", ascending=False)
df.to_csv("clean_numeric_data.csv", index=False)

print("\n CLEANED DATA:\n")
print(df.head())

print("\n  DATA INFO:")
print(df.info())

print("\n  MISSING VALUES:")
print(df.isnull().sum())

print("\n DUPLICATES:")
print(df.duplicated().sum())

print("\n FINAL FILE SAVED: clean_numeric_data.csv")