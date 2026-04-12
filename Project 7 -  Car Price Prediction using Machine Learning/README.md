# 🚗 Car Price Prediction using Machine Learning

## 🌐 Project Links
- 🔗 Dataset: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho  
- 💻 GitHub: https://github.com/mirzasameer2000/Machine-Learning-Projects/  
- 🌍 Portfolio: www.muhammadsameer.de  

---

## 👨‍🎓 Author
**Muhammad Sameer**  
🎓 Data Science Student – Arden University  

---

## 📌 Project Overview
This project predicts used car prices using Machine Learning models. The dataset includes features like engine specs, ownership, and fuel type.

---

## 📊 Dataset Description
- 2059 records
- Features: Price, Year, Kilometer, Fuel Type, Engine, Power, Torque, etc.

---

## 🧹 Data Preprocessing
- Missing values handled (median/mode)
- Feature engineering: Car_Age = Current Year - Year
- Converted Engine, Power, Torque to numeric
- One-hot encoding applied
- Feature scaling with StandardScaler
- Log transformation on target

---

## 🤖 Models Used
- Linear Regression
- Lasso Regression

---

## 📈 Model Performance

### Linear Regression
- R²: 0.91
- MAE: 0.208
- RMSE: 0.291

---

## 🔍 Lasso Alpha Analysis

| Alpha | R² | Explanation |
|------|----|------------|
| 0.001 | ~0.91 | No regularization |
| 0.01 | ~0.908 | Best balance |
| 0.1 | ~0.87 | Underfitting |

---

## 🧠 Final Model
Linear Regression selected due to best accuracy.

---

## 🚀 Run Project

git clone https://github.com/mirzasameer2000/Machine-Learning-Projects.git

pip install pandas numpy scikit-learn matplotlib seaborn

jupyter notebook

---

## ⭐ Support
Star ⭐ this repo if you like it!
