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

# STEP 7: Analyze negative billing amounts
print("\n===== NEGATIVE BILLING ANALYSIS =====")
negative_bills = cleaned[cleaned["Billing Amount"] < 0]
print("Negative billing records:", len(negative_bills))
print("\nSample negative billing records:")
print(
    negative_bills[
        ["Age", "Medical Condition", "Billing Amount"]].head(10) )
print("\nNegative billing summary:")
print(negative_bills["Billing Amount"].describe())

# STEP 8: Check details of negative billing records
print("\n===== NEGATIVE BILLING DETAILS =====")
negative_bills = cleaned[cleaned["Billing Amount"] < 0]
print("\nNegative billing by admission type:")
print(negative_bills["Admission Type"].value_counts())
print("\nNegative billing by insurance provider:")
print(negative_bills["Insurance Provider"].value_counts())
print("\nNegative billing by test results:")
print(negative_bills["Test Results"].value_counts())
print("\nBilling amount by admission type:")
print(
    negative_bills.groupby("Admission Type")["Billing Amount"]
    .agg(["count", "mean", "min", "max"]) )

## EDA 

# STEP 9: Exploratory Data Analysis (EDA)

# 1. Patient Age and Gender
print("\n===== AGE AND GENDER ANALYSIS =====")

print("\nAge Summary:")
print(cleaned["Age"].describe())

print("\nGender Distribution:")
print(cleaned["Gender"].value_counts())

print("\nGender Percentage:")
print((cleaned["Gender"].value_counts(normalize=True) * 100).round(2))

age_groups = pd.cut(
    cleaned["Age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["0-18", "19-35", "36-50", "51-65", "Above 65"])

print("\nAge Group Distribution:")
print(age_groups.value_counts().sort_index())

# 2. Medical Conditions
print("\n===== MEDICAL CONDITION ANALYSIS =====")

print(cleaned["Medical Condition"].value_counts())

print("\nAverage Age by Medical Condition:")
print(cleaned.groupby("Medical Condition")["Age"].mean().round(2))

# 3. Admission Type
print("\n===== ADMISSION TYPE ANALYSIS =====")

print(cleaned["Admission Type"].value_counts())

print("\nAverage Billing by Admission Type:")
print(
    cleaned.groupby("Admission Type")["Billing Amount"]
    .mean()
    .round(2))

# 4. Hospital Stay Duration
print("\n===== HOSPITAL STAY ANALYSIS =====")

admission_date = pd.to_datetime(
    cleaned["Date of Admission"], errors="coerce")
discharge_date = pd.to_datetime(
    cleaned["Discharge Date"], errors="coerce")

cleaned["Stay Days"] = (discharge_date - admission_date).dt.days

print(cleaned["Stay Days"].describe())

print("\nInvalid Stay Records:")
print((cleaned["Stay Days"] < 0).sum())

print("Missing Stay Values:", cleaned["Stay Days"].isnull().sum())
print("Zero-Day Stays:", (cleaned["Stay Days"] == 0).sum())

# 5. Billing Analysis
print("\n===== BILLING ANALYSIS =====")

print(cleaned["Billing Amount"].describe())

print("\nNegative Billing:", (cleaned["Billing Amount"] < 0).sum())
print("Zero Billing:", (cleaned["Billing Amount"] == 0).sum())

# 6. Insurance Provider Analysis
print("\n===== INSURANCE PROVIDER ANALYSIS =====")

print("\nPatient Records by Insurance Provider:")
print(cleaned["Insurance Provider"].value_counts())

print("\nBilling by Insurance Provider:")
print(
    cleaned.groupby("Insurance Provider")["Billing Amount"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# 7. Test Results
print("\n===== TEST RESULT ANALYSIS =====")

print(cleaned["Test Results"].value_counts())

print("\nTest Result Percentage:")
print((cleaned["Test Results"].value_counts(normalize=True) * 100).round(2))

# 8. Feature Relationships
print("\n===== FEATURE RELATIONSHIPS =====")

print("\nAverage Billing by Medical Condition:")
print(
    cleaned.groupby("Medical Condition")["Billing Amount"]
    .mean()
    .round(2))

print("\nAverage Stay by Admission Type:")
print(cleaned.groupby("Admission Type")["Stay Days"].mean().round(2))

print("\nNumerical Correlation:")
print(cleaned[["Age", "Billing Amount", "Stay Days"]].corr().round(3))

## STEP 10: Feature Engineering

print("\n===== FEATURE ENGINEERING =====")

# 1. Create Admission Month
cleaned["Admission Month"] = pd.to_datetime(
    cleaned["Date of Admission"], errors="coerce"
).dt.month

# 2. Reuse Stay Days calculated in EDA
cleaned["Length of Stay"] = cleaned["Stay Days"]

# 3. Create Age Groups
cleaned["Age Group"] = pd.cut(
    cleaned["Age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["0-18", "19-35", "36-50", "51-65", "Above 65"],
    include_lowest=True
)

# 4. Display the new features
print("\nNew Feature Sample:")
print(
    cleaned[
        ["Age", "Age Group", "Admission Month",
         "Stay Days", "Length of Stay"]
    ].head(10)
)

# 5. Validate new features
print("\nMissing Values in New Features:")
print(
    cleaned[
        ["Admission Month", "Length of Stay", "Age Group"]
    ].isnull().sum()
)

print("\nAdmission Month Values:")
print(sorted(cleaned["Admission Month"].dropna().unique()))

print("\nAge Group Counts:")
print(cleaned["Age Group"].value_counts().sort_index())
