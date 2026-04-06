
# 🍷 Wine Quality Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting the quality of red wine using machine learning techniques.  
The dataset used is from Kaggle:  
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009  

The goal is to classify wine samples as **Good Quality** or **Bad Quality** based on physicochemical properties.

---

## 📂 Dataset Information
- Total Samples: **1599**
- Features: **11 input features + 1 target variable**
- Target: `quality`

### Features:
- Fixed Acidity  
- Volatile Acidity  
- Citric Acid  
- Residual Sugar  
- Chlorides  
- Free Sulfur Dioxide  
- Total Sulfur Dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

---

## 🛠️ Technologies Used
- Python 🐍
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📊 Steps Performed in the Project

### 1. Data Loading
- Loaded dataset using Pandas (`read_csv`)
- Initial inspection using `.head()`, `.shape()`

### 2. Data Cleaning
- Checked for missing values
- No missing values found in dataset 

### 3. Exploratory Data Analysis (EDA)
Performed multiple visualizations:
- Count plot for wine quality distribution
- Bar plots:
  - Quality vs Volatile Acidity
  - Quality vs Citric Acid
  - Quality vs pH

### 4. Statistical Analysis
- Used `.describe()` to understand data distribution
- Calculated correlation matrix
- Visualized correlation using heatmap 

### 5. Feature Engineering
- Split dataset into:
  - Features (X)
  - Target (Y)

- Converted target into binary classification:
  - **1 → Good Quality (quality ≥ 7)**
  - **0 → Bad Quality (quality < 7)**

### 6. Train-Test Split
- Used `train_test_split`
- 80% Training, 20% Testing

### 7. Model Training
- Used **Random Forest Classifier**
- Trained model on training dataset

### 8. Model Evaluation
- Evaluated using **accuracy score**

---

## 🤖 Machine Learning Model
- Algorithm: **Random Forest Classifier**
- Reason:
  - Handles non-linearity well
  - Works effectively on tabular data
  - Provides good accuracy without heavy tuning

---

## 📈 Key Insights
- Alcohol has a strong positive correlation with wine quality
- Volatile acidity negatively impacts quality
- Dataset is slightly imbalanced (most wines fall in mid-quality range)

---

## 🚀 How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/mirzasameer2000/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run Notebook
Open Jupyter Notebook or Google Colab:
```bash
jupyter notebook
```

---

## 📁 Project Structure
```
Wine Quality Prediction/
│── winequality-red.csv
│── Wine_Quality_Prediction_using_Machine_Learning.ipynb
│── README.md
```

---

## 📌 Results
- Successfully built a classification model
- Achieved good prediction performance using Random Forest
- Identified key factors affecting wine quality

---

## 🔗 References
- Dataset: Kaggle Wine Quality Dataset
- Scikit-learn Documentation

---

## 👨‍💻 Author
**Muhammad Sameer**  
Data Science Student 
www.muhammadsameer.de 

---

## ⭐ Future Improvements
- Try other models (Logistic Regression, XGBoost, SVM)
- Hyperparameter tuning
- Deploy model using Flask or Django
- Add more datasets for generalization

---

## 📜 License
This project is for educational purposes.
