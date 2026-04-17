# Credit Card Fraud Detection using Machine Learning

A machine learning project that detects fraudulent credit card transactions using Logistic Regression. The dataset is highly imbalanced, so undersampling is applied to balance the classes before training.

---

## Dataset

**Source:** [Kaggle – Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Total Records:** 284,807 transactions
- **Features:** 31 columns (Time, V1–V28, Amount, Class)
- **Class Distribution:**
  - Legit (0): 284,315
  - Fraud (1): 492
- **Missing Values:** None
- **Memory Usage:** ~67.4 MB

> All features V1–V28 are PCA-transformed for privacy. Only `Time` and `Amount` are in original form.

---

## Project Structure

```
credit-card-fraud-detection/
│
├── creditcard.csv                  # Dataset (download from Kaggle)
├── Credit_Card_Fraud_Detection.ipynb
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## How It Works

### 1. Load and Explore Data

```python
import pandas as pd
import numpy as np

card_dataset = pd.read_csv('/content/creditcard.csv')

card_dataset.head()
card_dataset.shape         # (284807, 31)
card_dataset.info()
card_dataset.isnull().sum()  # No nulls
card_dataset['Class'].value_counts()
```

### 2. Separate Legit and Fraud Transactions

```python
legit = card_dataset[card_dataset.Class == 0]
fraud = card_dataset[card_dataset.Class == 1]

print(legit.shape)   # (284315, 31)
print(fraud.shape)   # (492, 31)

legit.Amount.describe()
fraud.Amount.describe()
```

**Mean Transaction Amounts:**
| Class | Mean Amount |
|-------|-------------|
| Legit | $88.29 |
| Fraud | $122.21 |

### 3. Handle Class Imbalance (Undersampling)

The dataset is heavily imbalanced (492 fraud vs 284,315 legit), so a random sample of 492 legit transactions is taken to balance the classes.

```python
legit_sample = legit.sample(n=492)
credit_card_dataset = pd.concat([legit_sample, fraud], axis=0)

credit_card_dataset['Class'].value_counts()
# Class 0: 492
# Class 1: 492
```

### 4. Prepare Features and Labels

```python
X = credit_card_dataset.drop(columns='Class', axis=1)
Y = credit_card_dataset['Class']
```

### 5. Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)
# (984, 30) (787, 30) (197, 30)
```

### 6. Train Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
```

> **Note:** A ConvergenceWarning appears because the default `max_iter=100` is not enough. You can fix this with `LogisticRegression(max_iter=1000)` or by scaling the data first.

### 7. Evaluate the Model

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Training accuracy
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data:', training_data_accuracy)
# 0.9454

# Test accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data:', test_data_accuracy)
# 0.9086
```

### 8. Confusion Matrix

```python
cm = confusion_matrix(Y_test, X_test_prediction)
print(cm)
# [[94  5]
#  [13 85]]
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()
```

### 9. Classification Report

```python
cp = classification_report(Y_test, X_test_prediction)
print(cp)
```

```
              precision    recall  f1-score   support

           0       0.88      0.95      0.91        99
           1       0.94      0.87      0.90        98

    accuracy                           0.91       197
   macro avg       0.91      0.91      0.91       197
weighted avg       0.91      0.91      0.91       197
```

---

## Results Summary

| Metric | Training | Test |
|--------|----------|------|
| Accuracy | 94.54% | 90.86% |
| Precision (Fraud) | — | 94% |
| Recall (Fraud) | — | 87% |
| F1-Score (Fraud) | — | 0.90 |

- **True Positives (Fraud caught):** 85
- **False Negatives (Fraud missed):** 13
- **False Positives (Legit flagged as Fraud):** 5

---

## Limitations & Possible Improvements

| Issue | Suggestion |
|-------|------------|
| Convergence warning | Use `max_iter=1000` or scale features with `StandardScaler` |
| Random undersampling loses data | Try SMOTE (oversampling minority class) |
| Only Logistic Regression used | Try Random Forest, XGBoost, or Neural Networks |
| No cross-validation | Use `StratifiedKFold` for more reliable evaluation |
| No feature importance | Analyze which V-features matter most for fraud |

---

## Key Concepts

- **PCA Features (V1–V28):** Original features are anonymized using Principal Component Analysis for privacy reasons.
- **Class Imbalance:** With only 0.17% fraud cases, accuracy alone is misleading — precision, recall, and F1 matter more.
- **Undersampling:** Randomly picking equal legit transactions keeps the model unbiased but throws away a lot of data.
- **Stratified Split:** Using `stratify=Y` ensures both train and test sets maintain the 50/50 class ratio.

---

## Author

**Muhammad Sameer**
- GitHub: [github.com/muhammadsameer](https://github.com/muhammadsameer)
- LinkedIn: [linkedin.com/in/muhammadsameer](https://linkedin.com/in/muhammadsameer)
- Portfolio: [muhammadsameer.de](https://www.muhammadsameer.de)
- Xing: [xing.com/profile/muhammadsameer](https://xing.com/profile/muhammadsameer)

---

## License

This project is for educational purposes. Dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) under the Open Database License (ODbL).
