# 🩺 Diabetes Prediction using Machine Learning

This project focuses on predicting whether a person is diabetic or not using a **Support Vector Machine (SVM)** classifier. It demonstrates a complete **end-to-end machine learning pipeline**, including data preprocessing, model training, evaluation, and building a predictive system.

---

## 🚀 Project Overview

Diabetes is a major global health issue. Early prediction can help in timely treatment and prevention.  
This project uses the **PIMA Diabetes Dataset** and applies machine learning techniques to build an accurate prediction model.

---

## 📂 Dataset

- **Source:** PIMA Indians Diabetes Dataset  
- **Total Records:** 768  
- **Features:** 8 input features  
  - Pregnancies  
  - Glucose  
  - Blood Pressure  
  - Skin Thickness  
  - Insulin  
  - BMI  
  - Diabetes Pedigree Function  
  - Age  
- **Target Variable:**  
  - `0` → Non-Diabetic  
  - `1` → Diabetic  

---

## ⚙️ Technologies Used

- Python 🐍  
- NumPy  
- Pandas  
- Scikit-learn  

---

## 🧠 Machine Learning Pipeline

### 1. Data Collection & Preprocessing
- Loaded dataset using Pandas
- Checked data shape, distribution, and summary statistics
- Separated features (`X`) and target (`Y`)

### 2. Data Standardization
- Applied `StandardScaler` to normalize feature values  
- Essential for SVM performance

### 3. Train-Test Split
- Split dataset into:
  - **80% Training**
  - **20% Testing**
- Used **stratified sampling** to maintain class balance

### 4. Model Training
- Algorithm: **Support Vector Machine (SVM)**
- Kernel used: **Linear Kernel**

### 5. Model Evaluation
- Training Accuracy: **78.66%**
- Testing Accuracy: **77.27%**

---

## 📊 Results

| Metric              | Score     |
|--------------------|----------|
| Training Accuracy  | 78.66%   |
| Testing Accuracy   | 77.27%   |

---

## 🔮 Predictive System

A simple predictive system is implemented that:

1. Takes user input (health parameters)
2. Standardizes the input
3. Uses the trained SVM model to predict diabetes

### Example Input:
```python
input_data = (0,180,66,39,0,42,1.893,25)
```

### Output:
```
The person is diabetic
```

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/mirzasameer2000/Machine-Learning-Projects.git
cd diabetes-prediction
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn
```

### 3. Run the Project
```bash
python main.py
```

---

## 📌 Key Learnings

- Importance of **data preprocessing and scaling**
- Understanding **SVM algorithm behavior**
- Handling **imbalanced datasets using stratification**
- Building a **real-world predictive system**

---

## 🔮 Future Improvements

- Try advanced models (Random Forest, XGBoost, Neural Networks)
- Perform hyperparameter tuning
- Add a web interface (Streamlit / Flask)
- Deploy the model

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 📧 Contact

**Muhammad Sameer**  
Data Science Student at Arden University Berlin
