# Healthcare Analytics & Predictive Modeling Using Machine Learning

## About the Project

This is my Data Science and Machine Learning project developed using Python. The main purpose of this project is to analyze healthcare data, understand patient-related patterns, and compare different Machine Learning models for predicting patient test results.

## Project Objectives

- Clean and prepare the healthcare dataset.
- Analyze patient age, gender, and medical conditions.
- Study hospital admission types and billing amounts.
- Explore test result patterns using data visualization.
- Create new features from admission and discharge dates.
- Build and compare four Machine Learning classification models.

## Dataset

The project uses the Healthcare Dataset available on Kaggle.

- **Dataset Source:** [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- **EDA Reference:** [Healthcare Dataset EDA Notebook](https://www.kaggle.com/code/hainescity/healthcare-dataset-eda/input)

The dataset includes information about patient demographics, medical conditions, hospital admissions, insurance providers, medications, billing amounts, and test results.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn

## Project Workflow

### 1. Data Cleaning

Checked for missing values and duplicate records and prepared the dataset for analysis.

### 2. Exploratory Data Analysis (EDA)

Analyzed patient demographics, medical conditions, admission types, billing amounts, and test results.

### 3. Feature Engineering

Created new features such as admission month and length of hospital stay.

### 4. Data Visualization

Created graphs to understand the distribution of data and identify patterns.

### 5. Machine Learning

Implemented the following classification models to predict the `Test Results` category:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### 6. Model Evaluation

Compared model performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## How to Run the Project

### Prerequisites

- Python 3.x
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/ahsan76548/Healthcare-Analytics-Predictive-Modeling.git
```

### Step 2: Open the Project Folder

```bash
cd Healthcare-Analytics-Predictive-Modeling
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 4: Download the Dataset

Download the dataset from the [Kaggle Dataset Page](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) and place the CSV file in the project folder.

Make sure the filename and file path match the ones used in the Python code.

### Step 5: Run the Python File

```bash
python healthcare_project.py
```

The program will perform the analysis, display visualizations, train and evaluate the Machine Learning models, and save the model comparison results if the corresponding code is included.

## Results

The four Machine Learning models are evaluated and compared using different performance metrics. The model comparison results can be found in `model_comparison.csv` after running the project.

Actual performance depends on the dataset, selected features, and model settings.

## Conclusion

This project helped me gain practical experience in data cleaning, exploratory data analysis, feature engineering, data visualization, and Machine Learning model evaluation using Python.

## Author

**MD Ahsan Ali**

Data Science Student

## Disclaimer

This project is for educational purposes only and is not intended for medical diagnosis or clinical decision-making.
