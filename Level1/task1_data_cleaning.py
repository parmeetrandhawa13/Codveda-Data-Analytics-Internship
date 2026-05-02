"""
Codveda Technology - Data Analytics Internship
Level 1 | Task 1: Data Cleaning and Preprocessing
Dataset: House Prediction Data Set (Boston Housing)
"""

import pandas as pd
import numpy as np
# ── 1. Load Dataset 
print("=" * 60)
print("TASK 1: DATA CLEANING AND PREPROCESSING")
print("=" * 60)

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

df = pd.read_csv(
    "4__house_Prediction_Data_Set.csv",
    header=None,
    sep=r'\s+',
    names=columns
)

print(f"\n✅ Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n── Column Names ──")
print(columns)
# ── 2. Initial Inspection 
print("\n── First 5 Rows ──")
print(df.head())

print("\n── Data Types ──")
print(df.dtypes)

print("\n── Basic Statistics ──")
print(df.describe().round(2))
# ── 3. Inject Artificial Issues for Demonstration 
print("\n── Injecting missing values & duplicates for demonstration ──")
np.random.seed(42)

# Add missing values (~5% per column)
for col in ["RM", "AGE", "LSTAT", "MEDV", "CRIM"]:
    idx = np.random.choice(df.index, size=25, replace=False)
    df.loc[idx, col] = np.nan

# Add duplicate rows
df = pd.concat([df, df.sample(10, random_state=42)], ignore_index=True)

# Add inconsistent formats in a categorical-style column
df["CHAS"] = df["CHAS"].astype(str)
df.loc[df.sample(15, random_state=1).index, "CHAS"] = " 1 "   # extra spaces
df.loc[df.sample(10, random_state=2).index, "CHAS"] = "Yes"   # inconsistent label

print(f"   Missing values added. Duplicates added.")
# ── 4. Identify Issues 
print("\n── Missing Values Per Column ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_report[missing_report["Missing Count"] > 0])

print(f"\n── Duplicate Rows: {df.duplicated().sum()} ──")

print(f"\n── Unique values in CHAS (inconsistent): {df['CHAS'].unique()} ──")
# ── 5. Handle Missing Values 
print("\n── Handling Missing Values ──")

# Numerical columns: fill with median (robust to outliers)
numerical_cols = ["RM", "AGE", "LSTAT", "MEDV", "CRIM"]
for col in numerical_cols:
    median_val = df[col].median()
    before = df[col].isnull().sum()
    df[col] = df[col].fillna(median_val)
    print(f"   {col}: {before} missing → filled with median ({median_val:.2f})")
# ── 6. Remove Duplicates 
print("\n── Removing Duplicate Rows ──")
before_dedup = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"   Rows before: {before_dedup} → After: {len(df)} (removed {before_dedup - len(df)})")
# ── 7. Standardize Inconsistent Formats
print("\n── Standardizing CHAS Column ──")
print(f"   Before: {df['CHAS'].unique()}")
df["CHAS"] = df["CHAS"].str.strip()                         # remove spaces
df["CHAS"] = df["CHAS"].replace({"Yes": "1", "No": "0"})   # unify labels
df["CHAS"] = df["CHAS"].astype(int)                         # convert to int
print(f"   After:  {df['CHAS'].unique()}")
# ── 8. Final Report 
print("\n── Final Dataset Info ──")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Total Missing Values: {df.isnull().sum().sum()}")
print(f"   Total Duplicates:     {df.duplicated().sum()}")
print("\n── Sample of Cleaned Data ──")
print(df.head())
# ── 9. Save Cleaned Dataset 
output_path = "cleaned_house_data.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_path}")
print("=" * 60)
print("TASK 1 COMPLETE ✅")
print("=" * 60)
