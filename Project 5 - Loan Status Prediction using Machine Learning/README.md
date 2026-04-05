# 🏦 Loan Status Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting whether a loan application will be approved or rejected using Machine Learning techniques. The goal is to build a predictive model that can assist financial institutions in making faster and more accurate loan approval decisions.

---

## 📂 Dataset
- Source: Kaggle  
- Dataset Link: https://www.kaggle.com/datasets/ninzaami/loan-predication  

The dataset contains 614 rows and 13 columns, including applicant information such as income, education, marital status, and loan details.

---

## 🧠 Project Workflow

### 1. Data Loading & Exploration
- Imported dataset using Pandas
- Checked dataset shape and structure
- Used .head(), .describe(), and .info() for initial exploration
- Identified missing values across multiple columns  

---

### 2. Data Cleaning
- Handled missing values by dropping null rows
- Reduced dataset size from 614 → 480 rows after cleaning  
- Converted target variable:
  - Loan_Status: Y → 1, N → 0

---

### 3. Feature Engineering
- Cleaned categorical inconsistencies:
  - Replaced "3+" in Dependents → 4
- Converted categorical variables into numerical values:
  - Gender: Male = 1, Female = 0
  - Married: Yes = 1, No = 0
  - Self_Employed: Yes = 1, No = 0
  - Property_Area:
    - Rural = 0
    - Semiurban = 1
    - Urban = 2
  - Education:
    - Graduate = 1
    - Not Graduate = 0  

---

### 4. Data Visualization 📊
Used Seaborn to analyze relationships between features and loan status:
- Education vs Loan Status
- Marital Status vs Loan Status
- Gender vs Loan Status
- Employment Status vs Loan Status
- Property Area vs Loan Status  

---

### 5. Feature & Target Separation
- Features (X): All columns except Loan_ID and Loan_Status
- Target (Y): Loan_Status

---

### 6. Model Building 🤖
- Used Support Vector Machine (SVM)
- Split data using train_test_split

---

### 7. Model Evaluation
- Evaluated using accuracy score

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Seaborn & Matplotlib
- Scikit-learn

---

## 📈 Key Learnings
- Data preprocessing
- Handling categorical variables
- Exploratory Data Analysis (EDA)
- Machine Learning model building

---

## 🚀 Future Improvements
- Random Forest / XGBoost
- Hyperparameter tuning
- Cross-validation
- Deployment

---

## 👨‍💻 Author
Muhammad Sameer  
Data Science Student at Arden University Berlin  

- Portfolio: www.muhammadsameer.de  
- GitHub: https://github.com/mirzasameer2000/Machine-Learning-Projects  

---

## ⭐ Show Your Support
If you like this project, please star the repository!
