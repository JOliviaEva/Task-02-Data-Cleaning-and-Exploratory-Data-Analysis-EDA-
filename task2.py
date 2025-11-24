# -----------------------------
# Task 02: Data Cleaning + EDA
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# 1. Load the Dataset
# -----------------------------------------
file_path = r"D:\Prodigy\task2\data-task2.csv" 
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print(df.head())
print("\nShape of dataset:", df.shape)

# -----------------------------------------
# 2. Basic Information
# -----------------------------------------
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all'))

# -----------------------------------------
# 3. Check Missing Values
# -----------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# -----------------------------------------
# 4. Handle Missing Data
# -----------------------------------------

# Example: Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Example: Fill categorical missing values with mode
categorical_cols = df.select_dtypes(exclude=["number"]).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------------------
# 5. Exploratory Data Analysis (EDA)
# -----------------------------------------

# ---- Distribution of a numeric variable ----
if len(numeric_cols) > 0:
    num = numeric_cols[0]
    plt.figure(figsize=(8,5))
    plt.hist(df[num], bins=20)
    plt.title(f"Distribution of {num}")
    plt.xlabel(num)
    plt.ylabel("Frequency")
    plt.show()

# ---- Countplot of a categorical variable ----
if len(categorical_cols) > 0:
    cat = categorical_cols[0]
    plt.figure(figsize=(8,5))
    sns.countplot(x=df[cat])
    plt.title(f"Count Plot of {cat}")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------------------
# 6. Correlation Heatmap (Numeric Only)
# -----------------------------------------

numeric_df = df.select_dtypes(include=["number"])

if numeric_df.shape[1] > 1:  # only plot if more than 1 numeric col
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Variables)")
    plt.show()
else:
    print("Not enough numeric columns for correlation heatmap.")

# -----------------------------------------
# 7. Relationship Analysis
# -----------------------------------------

# Example: numeric vs categorical
if len(numeric_cols) > 0 and len(categorical_cols) > 0:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[categorical_cols[0]], y=df[numeric_cols[0]])
    plt.title(f"{numeric_cols[0]} by {categorical_cols[0]}")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------------------
# 8. Final Cleaned Dataset Preview
# -----------------------------------------
print("\nFinal Cleaned Dataset Head:")
print(df.head())

print("\nTASK 02 COMPLETED SUCCESSFULLY 🎉")
