# Healthcare Analytics & Predictive Modeling Using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

#  Load the dataset
data = pd.read_csv("healthcare_dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nStatistical Summary:")
print(data.describe())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

#  Data cleaning
data.columns = data.columns.str.strip()

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()

data = data.drop_duplicates()

data["Age"] = pd.to_numeric(data["Age"], errors="coerce")
data["Billing Amount"] = pd.to_numeric(
    data["Billing Amount"], errors="coerce")

data["Date of Admission"] = pd.to_datetime(
    data["Date of Admission"], errors="coerce")
data["Discharge Date"] = pd.to_datetime(
    data["Discharge Date"], errors="coerce")

data = data.dropna(subset=[
    "Age", "Billing Amount", "Date of Admission",
    "Discharge Date", "Test Results"])

data = data[
    (data["Age"].between(0, 120)) &
    (data["Billing Amount"] >= 0) &
    (data["Discharge Date"] >= data["Date of Admission"])
].copy()

print("\nShape after cleaning:", data.shape)
print("\nMissing values after cleaning:")
print(data.isnull().sum())

#  Feature engineering
data["Admission Month"] = data["Date of Admission"].dt.month
data["Length of Stay"] = (
    data["Discharge Date"] - data["Date of Admission"]).dt.days

print("\nLength of stay summary:")
print(data["Length of Stay"].describe())

#  EDA - (Exploratory Data Analysis)
# Objective 1: Age and gender
print("\nObjective 1: Age by gender")
print(data.groupby("Gender")["Age"].agg(["count", "mean", "median"]))

plt.figure(figsize=(7, 4))
plt.hist(data["Age"], bins=15, edgecolor="black")
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

#  Objective 2: Medical conditions
print("\nObjective 2: Medical condition counts")
print(data["Medical Condition"].value_counts())

plt.figure(figsize=(8, 4))
sns.countplot(data=data, x="Medical Condition",
              order=data["Medical Condition"].value_counts().index)
plt.title("Medical Condition Distribution")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#  Objective 3: Billing analysis
print("\nObjective 4: Average billing by medical condition")
print(data.groupby("Medical Condition")["Billing Amount"].mean().sort_values(
    ascending=False))

plt.figure(figsize=(8, 4))
sns.barplot(data=data, x="Medical Condition", y="Billing Amount",errorbar=None)
plt.title("Average Billing by Medical Condition")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#  Objective 4: Test results
print("\nObjective 5: Test result counts")
print(data["Test Results"].value_counts())

print("\nTest results by medical condition (%)")
print((pd.crosstab(data["Medical Condition"], data["Test Results"],
                   normalize="index") * 100).round(2))

plt.figure(figsize=(7, 4))
sns.countplot(data=data, x="Test Results")
plt.title("Test Results Distribution")
plt.tight_layout()
plt.show()

#  Objective 5: Length of Stay Analysis
print("\nLength of Stay Counts:")
print(data["Length of Stay"].value_counts().sort_index())

plt.figure(figsize=(7, 4))
sns.histplot(data=data, x="Length of Stay", bins=20)
plt.title("Hospital Length of Stay")
plt.xlabel("Length of Stay (Days)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()


# STEP 9: Machine Learning
# Target: predict Test Results (classification)
# Select input features
X = data[[
    "Age", "Admission Month", "Gender", "Blood Type",
    "Medical Condition", "Insurance Provider",
    "Admission Type", "Medication"
]]

y = data["Test Results"]

# Convert text categories into numbers
X = pd.get_dummies(X, drop_first=False)

# Split data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=42, stratify=y)

# Scale features for Logistic Regression and SVM
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Four machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42
    ),
    "SVM": SVC()
}

results = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results.append([name, accuracy, precision, recall, f1])

    print("\nModel:", name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", round(accuracy, 4))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
# Confusion Matrix Graph
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix - " + name)
plt.tight_layout()
plt.show()

# Compare all models
result_table = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

result_table = result_table.sort_values(
    by="F1 Score", ascending=False
)

print("\nModel Comparison:")
print(result_table.to_string(index=False))

# Graph 1: Accuracy Comparison
plt.figure(figsize=(8, 4))
plt.bar(result_table["Model"], result_table["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Graph 2: F1 Score Comparison
plt.figure(figsize=(8, 4))
plt.bar(result_table["Model"], result_table["F1 Score"])
plt.title("Model F1 Score Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Macro F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Best model based on Macro F1 Score
best_model = result_table.iloc[0]

print("\nBest Model:", best_model["Model"])
print("Best Macro F1 Score:", round(best_model["F1 Score"], 4))

result_table.to_csv("model_comparison.csv", index=False)

print("\nCompleted.")

