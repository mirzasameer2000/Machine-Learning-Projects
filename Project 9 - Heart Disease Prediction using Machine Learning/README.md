# ❤️ Heart Disease Prediction using Machine Learning

A binary classification project that predicts whether a person has **heart disease** using clinical health indicators and **Logistic Regression**, achieving **81.97% accuracy** on unseen test data.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early and accurate prediction using patient clinical data can significantly improve treatment outcomes. This project builds a machine learning model trained on real clinical data that classifies a patient as either having heart disease or not — using 13 medical features.

| Metric | Training Data | Test Data |
|---|---|---|
| Accuracy | **85.12%** | **81.97%** |
| Precision (avg) | — | **0.82** |
| Recall (avg) | — | **0.82** |
| F1-Score (avg) | — | **0.82** |

> The small gap between training (85.12%) and test (81.97%) accuracy indicates the model generalises reasonably well with **minimal overfitting**.

---

## 📊 Dataset

- **Source:** Dataset (`heart_disease_data.csv`) is available in the [GitHub repository](https://github.com/mirzasameer2000/Machine-Learning-Projects/)
- **Size:** 303 rows × 14 columns
- **No missing values**
- **Target classes:** `1` = Heart Disease Present, `0` = No Heart Disease

### Class Distribution

| Class | Count |
|---|---|
| Defective Heart (1) | 165 |
| Healthy Heart (0) | 138 |

> Slightly imbalanced but close enough that **stratified splitting** handles it well — which is used in this project.

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia type (0–3) |
| `target` | **Target variable** — 1 = disease, 0 = no disease |

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:**
  - `pandas`, `numpy` — data loading and manipulation
  - `seaborn`, `matplotlib` — visualisation
  - `scikit-learn` — model training, evaluation, and metrics

---

## 🧠 Model

**Algorithm:** `LogisticRegression` from scikit-learn

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

- **Train/Test Split:** 80% / 20% with `stratify=Y` to preserve class balance
- **Training samples:** 242 | **Test samples:** 61
- **Features used:** All 13 clinical features
- **Target:** `target` (binary: 0 or 1)

> **Note:** A convergence warning appears during training (`max_iter` reached). This can be fixed by adding `max_iter=1000` or scaling the features with `StandardScaler` — both are listed in the improvements section below.

---

## 📈 Results

### Accuracy
```
Training Accuracy:  0.8512  (85.12%)
Test Accuracy:      0.8197  (81.97%)
```

### Confusion Matrix (Test Set)
```
[[23  6]
 [ 5 27]]
```

| | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | 23 ✅ | 6 ❌ |
| **Actual: Disease** | 5 ❌ | 27 ✅ |

- **True Positives (Disease correctly detected):** 27
- **True Negatives (Healthy correctly identified):** 23
- **False Negatives (Missed disease cases):** 5 ← most important to minimise in healthcare

### Classification Report
```
              precision  recall  f1-score  support
           0       0.82    0.79      0.81       29
           1       0.82    0.84      0.83       32
    accuracy                         0.82       61
   macro avg       0.82    0.82      0.82       61
weighted avg       0.82    0.82      0.82       61
```

---

## 🔮 Single Patient Prediction

The model also supports real-time prediction for individual patients:

```python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
```

**Output for the above sample:** `The Person does not have a Heart Disease`

---

## 🗂️ Project Structure

```
heart-disease-prediction/
├── Heart_Disease_Prediction.ipynb    # Main Colab notebook
├── heart_disease_data.csv            # Dataset
└── README.md                         # Project documentation
```

---

## ▶️ How to Run

1. Open the notebook in Google Colab
2. Upload `heart_disease_data.csv` to the Colab session
3. Run all cells in order

```bash
# Or run locally
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook Heart_Disease_Prediction.ipynb
```

---

## 🔍 Key Steps

1. **Data Loading & Inspection** — `head()`, `shape`, `info()`, `describe()`, null checks
2. **EDA** — Class distribution, feature statistics
3. **Feature/Target Split** — `X` = 13 features, `Y` = target column
4. **Stratified Train/Test Split** — 80/20 split preserving class ratio
5. **Model Training** — Logistic Regression
6. **Evaluation** — Accuracy, confusion matrix, classification report
7. **Single Prediction** — Real-time inference on new patient data
8. **Visualisation** — Training vs test accuracy bar chart

---

## 💡 Insights

- The model achieves **balanced precision and recall (0.82)** for both classes — important in medical diagnosis where missing a disease case (false negative) is costly
- **5 false negatives** out of 61 test samples — the model missed 5 actual disease cases, which highlights the need for further improvement in recall for class 1
- **Stratified splitting** was correctly applied, ensuring the train/test sets reflect the original class imbalance (~54% disease, ~46% healthy)
- The **small train-test gap (~3%)** suggests the model is not significantly overfitting despite the convergence warning

---

## 🚀 Possible Improvements

- Fix convergence warning by adding `max_iter=1000` or using `StandardScaler` to normalise features
- Try other classifiers: **Random Forest, SVM, XGBoost** for comparison
- Apply **k-fold cross-validation** for more robust accuracy estimates
- Add **feature importance / coefficients** plot to understand which clinical factors matter most
- Use **SMOTE** or class weighting to further improve recall for disease class
- Deploy as a **Streamlit web app** for real-time patient risk screening

---

## 👤 Author

**Muhammad Sameer**
MSc Data Science — Arden University Berlin

🌐 [muhammadsameer.de](https://www.muhammadsameer.de)
💼 [LinkedIn](https://linkedin.com/in/mirzasameerbaig99)
🐙 [GitHub — ML Projects](https://github.com/mirzasameer2000/Machine-Learning-Projects/)
🔗 [Xing](https://xing.com/Muhammad_Sameer033677)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
