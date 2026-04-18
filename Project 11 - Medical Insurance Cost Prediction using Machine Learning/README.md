# Medical Insurance Cost Prediction using Machine Learning

A machine learning project to predict medical insurance charges based on patient data like age, BMI, smoking status, and region. I built and compared two models — Linear Regression and XGBoost — to see which one does better on this problem.

Dataset from Kaggle: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

\---

## What's in the Dataset

The dataset has 1338 rows and 7 columns. No missing values.

|Column|Type|Description|
|-|-|-|
|age|int|Age of the patient|
|sex|object|male / female|
|bmi|float|Body mass index|
|children|int|Number of kids covered by insurance|
|smoker|object|yes / no|
|region|object|southwest, southeast, northwest, northeast|
|charges|float|Medical insurance cost (target)|

Quick stats on the numeric columns:

* Age: mean \~39, range 18–64
* BMI: mean \~30.6, range 16–53
* Charges: mean \~$13,270, max \~$63,770

\---

## Project Structure

```
medical-insurance-cost-prediction/
│
├── insurance.csv                        # raw dataset
├── insurance\_dataset\_preprocessed.csv  # after encoding
├── Medical\_Insurance\_Cost\_Prediction\_using\_Machine\_Learning.ipynb
└── README.md
```

\---

## Steps I Followed

### 1\. EDA (Exploratory Data Analysis)

Plotted distributions for:

* Sex (roughly balanced — 676 male, 662 female)
* Age (mostly spread between 18–64)
* BMI (close to normal distribution, centered around 30)
* Children (most people have 0 or 1 child)
* Smoker (only 274 out of 1338 are smokers)
* Region (fairly even across all 4 regions)
* Charges (right-skewed — most charges are low, few are very high)

### 2\. Preprocessing

No missing values so nothing to impute. Just label encoding for the categorical columns:

```python
insurance\_dataset.replace({'male': 0, 'female': 1}, inplace=True)
insurance\_dataset.replace({'yes': 0, 'no': 1}, inplace=True)
insurance\_dataset.replace({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}, inplace=True)
```

### 3\. Train/Test Split

```python
X = insurance\_dataset.drop(columns='charges', axis=1)
Y = insurance\_dataset\['charges']

X\_train, X\_test, Y\_train, Y\_test = train\_test\_split(X, Y, test\_size=0.2, random\_state=2)
# X: (1338, 6) | X\_train: (1070, 6) | X\_test: (268, 6)
```

\---

## Models

### Linear Regression

```python
from sklearn.linear\_model import LinearRegression

model = LinearRegression()
model.fit(X\_train, Y\_train)
```

|Metric|Train|Test|
|-|-|-|
|R²|0.752|0.744|
|MAE|4140.03|4285.22|
|MSE|36,104,122|38,364,832|
|RMSE|6008.67|6193.94|

### XGBoost Regressor

```python
import xgboost as xgb

xgb\_model = xgb.XGBRegressor(
    n\_estimators=100,
    learning\_rate=0.1,
    max\_depth=5,
    objective='reg:squarederror'
)
xgb\_model.fit(X\_train, Y\_train)
```

|Metric|Train|Test|
|-|-|-|
|R²|0.943|0.857|
|MAE|1539.76|2530.91|
|MSE|8,238,167|21,434,764|
|RMSE|2870.22|4629.77|

\---

## Results

XGBoost clearly beats Linear Regression. R² of 0.857 on test vs 0.744 — and MAE dropped from \~$4,285 to \~$2,531. The gap between XGBoost train (0.943) and test (0.857) shows some overfitting but it's not too bad for this dataset size.

Linear Regression is decent as a baseline but the charges distribution is right-skewed, so a tree-based model handles it better.

\---

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Then just open the notebook and run all cells. The dataset path is `/content/insurance.csv` (Colab) — update it to your local path if needed.

\---

## Libraries Used

* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* xgboost

\---

## Notes

* `sns.distplot()` is deprecated — use `sns.histplot()` or `sns.displot()` for newer seaborn versions
* `inplace=True` in `.replace()` will throw a FutureWarning in newer pandas — can switch to `insurance\_dataset = insurance\_dataset.replace(...)` instead

