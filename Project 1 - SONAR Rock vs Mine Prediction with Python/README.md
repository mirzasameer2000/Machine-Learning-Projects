# 🚀 Sonar Data Classification using Logistic Regression

This project focuses on building a **Machine Learning model** to classify objects detected by sonar signals as either a **Mine (M)** or a **Rock (R)**. It demonstrates a complete ML workflow, from data preprocessing to model evaluation and prediction.

---

## 📌 Project Overview

The goal of this project is to develop a predictive system using **Logistic Regression** that can accurately classify sonar signal data.

---

## 📂 Dataset Information

- **Dataset:** `sonar_data.csv`
- **Total Samples:** 208
- **Total Features:** 61  
  - 60 numerical sonar signal features  
  - 1 target label (M = Mine, R = Rock)

### 🎯 Target Distribution
- Mines (M): 111  
- Rocks (R): 97  

---

## ⚙️ Workflow

### 1️⃣ Data Collection & Preprocessing
- Loaded dataset using **Pandas**
- Explored dataset shape: `(208, 61)`
- Analyzed statistical measures
- Separated:
  - Features → `X`
  - Target → `Y`

---

### 2️⃣ Data Splitting
- Used `train_test_split` from **Scikit-learn**
- **Test Size:** 10%
- **Stratification:** Enabled (`stratify=Y`) to maintain class balance

| Dataset       | Samples |
|--------------|--------|
| Training Set | 187    |
| Test Set     | 21     |

---

### 3️⃣ Model Training
- Algorithm used: **Logistic Regression**
- Suitable for binary classification tasks
- Trained model on the training dataset

---

### 4️⃣ Model Evaluation

| Metric                | Score     |
|---------------------|----------|
| Training Accuracy    | 83.42%   |
| Testing Accuracy     | 76.19%   |

📊 The model shows good generalization on unseen data.

---

### 5️⃣ Predictive System
- Built a simple prediction system
- Takes new sonar input data
- Outputs classification:
  - **Mine** 🧨
  - **Rock** 🪨

✔️ Example prediction successfully classified an object as **Mine**

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas
- NumPy
- Scikit-learn

---

## 📈 Key Learnings
- Data preprocessing and feature handling
- Importance of stratified sampling
- Logistic Regression for binary classification
- Model evaluation and accuracy comparison
- Building real-world predictive systems

---

## 🚀 Future Improvements
- Try advanced models (SVM, Random Forest, Neural Networks)
- Hyperparameter tuning
- Cross-validation
- Deployment as a web app (Flask/Django)

---

## 🧠 Conclusion
This project highlights how a simple yet powerful algorithm like **Logistic Regression** can effectively solve real-world classification problems.

---

## 📬 Connect with Me
If you like this project, feel free to ⭐ the repo and connect with me!

---

#MachineLearning #DataScience #Python #LogisticRegression #PredictiveAnalytics
