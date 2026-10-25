import numpy as np
import pandas as pd

# STEP 1: Load the original dataset
data = pd.read_csv("healthcare_dataset.csv")

# STEP 2: Check the dataset
print("===== DATASET INFORMATION =====")

print("\nDataset Shape:")
print(data.shape)
print("\nFirst 5 Rows:")
print(data.head())
print("\nColumn Names:")
print(data.columns.tolist())
print("\nData Types:")
print(data.dtypes)
print("\nStatistical Summary:")
print(data.describe(include="all"))


# STEP 3: Check data quality
print("\n===== DATA QUALITY CHECKS =====")

print("\nMissing Values:")
print(data.isnull().sum())
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Check text formatting
print("\nText Formatting Check:")
for col in data.select_dtypes(include="object").columns:
    print(col, "Extra spaces:", data[col].str.strip().ne(data[col]).sum())

# Check categorical values
print("\nCategory Values:")
for col in ["Gender", "Blood Type", "Medical Condition",
            "Admission Type", "Medication", "Test Results"]:
    print(col, ":", data[col].unique())

# Check date values
admission_date = pd.to_datetime(
    data["Date of Admission"], errors="coerce")
discharge_date = pd.to_datetime(
    data["Discharge Date"], errors="coerce")

print("Invalid Admission Dates:", admission_date.isnull().sum())
print("Invalid Discharge Dates:", discharge_date.isnull().sum())
print("Discharge Before Admission:", (discharge_date < admission_date).sum())

# Check numerical values
print("\nNumerical Checks:")
print("Negative Age:", (data["Age"] < 0).sum())
print("Negative Billing:", (data["Billing Amount"] < 0).sum())
print("Zero Billing:", (data["Billing Amount"] == 0).sum())

# STEP 4: Clean the dataset
cleaned = data.copy()

# Remove duplicate rows
cleaned = cleaned.drop_duplicates()

# Remove extra spaces from text columns
for col in cleaned.select_dtypes(include="object").columns:
    cleaned[col] = cleaned[col].str.strip()

# Standardize patient names
cleaned["Name"] = cleaned["Name"].str.title()

# STEP 5: Verify cleaning results
print("\n===== DATA CLEANING RESULTS =====")

print("Original Rows:", len(data))
print("Rows After Cleaning:", len(cleaned))
print("Duplicates Remaining:", cleaned.duplicated().sum())

# STEP 6: Save the cleaned dataset
cleaned.to_csv("healthcare_cleaned.csv", index=False)

print("\nFinal Dataset Shape:")
print(cleaned.shape)

print("\nCleaned dataset saved successfully!")

