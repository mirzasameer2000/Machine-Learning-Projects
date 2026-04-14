# 🥇 Gold Price Prediction using Machine Learning

A machine learning project that predicts **GLD (Gold ETF) prices** using financial market indicators and a **Random Forest Regressor**, achieving an R² score of **0.9894** on unseen test data.

---

## 📌 Project Overview

Gold prices are influenced by a range of macroeconomic factors including stock market performance, oil prices, silver prices, and currency exchange rates. This project uses historical data from these indicators to train a predictive model that estimates the GLD price with high accuracy.

| Metric | Score |
|---|---|
| R² Score (Test) | **0.9894** |
| Mean Squared Error | 5.576 |
| Mean Absolute Error | **1.323** |

> The model explains **98.94% of the variance** in gold prices on completely unseen test data — demonstrating strong generalisation with no overfitting.

---

## 📊 Dataset

- **Source:** [Gold Price Data — Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data)
- **Size:** 2,290 rows × 6 columns
- **Date Range:** January 2008 – May 2018
- **No missing values**

### Features

| Column | Description |
|---|---|
| `SPX` | S&P 500 Index price |
| `GLD` | Gold ETF price (**target variable**) |
| `USO` | United States Oil Fund price |
| `SLV` | Silver ETF price |
| `EUR/USD` | Euro to US Dollar exchange rate |

### Key Correlations with GLD

| Feature | Correlation |
|---|---|
| SLV | **+0.87** (strong positive) |
| USO | -0.19 |
| SPX | +0.05 |
| EUR/USD | -0.02 |

> Silver (`SLV`) is the strongest predictor of gold prices, which aligns with real-world commodity market behaviour.

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Environment:** Google Colab
- **Libraries:**
  - `pandas`, `numpy` — data loading and manipulation
  - `seaborn`, `matplotlib` — EDA and visualisation
  - `scikit-learn` — model training, splitting, and evaluation

---

## 🧠 Model

**Algorithm:** `RandomForestRegressor` from scikit-learn

```python
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
```

- **Train/Test Split:** 80% training / 20% test (`random_state` fixed for reproducibility)
- **Features used:** SPX, USO, SLV, EUR/USD
- **Target:** GLD (Gold ETF price)

---

## 📈 Results

```
R² Score on Test data:          0.9894
Mean Squared Error on Test data: 5.5760
Mean Absolute Error on Test data: 1.3228
```

The actual vs predicted price plot shows the green (predicted) line closely tracking the blue (actual) line across all 458 test samples — confirming the model generalises well across varying market conditions.

---

## 🗂️ Project Structure

```
gold-price-prediction/
├── Gold_Price_Prediction_using_Machine_Learning.ipynb   # Main Colab notebook
├── gld_price_data.csv            # Dataset (download from Kaggle)
└── README.md                     # Project documentation
```

---

## ▶️ How to Run

1. Clone this repository or open the notebook in Google Colab
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data) and upload it to Colab as `gld_price_data.csv`
3. Run all cells in order

```bash
# Or run locally
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook Gold_Price_Prediction_using_Machine_Learning.ipynb
```

---

## 🔍 Key Steps

1. **Data Loading & Inspection** — `head()`, `tail()`, `info()`, `describe()`, null checks
2. **Exploratory Data Analysis** — Correlation heatmap, GLD distribution plot
3. **Feature Engineering** — Dropped `Date` and `GLD` from features; `GLD` used as target
4. **Train/Test Split** — 80/20 split with fixed random state
5. **Model Training** — Random Forest with 100 estimators
6. **Evaluation** — R², MSE, MAE on test set
7. **Visualisation** — Actual vs Predicted price line chart

---

## 💡 Insights

- **Silver is the strongest predictor** of gold prices (correlation = 0.87), reflecting the historical co-movement of precious metals
- **Oil and EUR/USD** have weak or negative correlation with GLD, suggesting they add minor but useful signal in the ensemble model
- The **bimodal GLD distribution** (peaks around ~90 and ~125) reflects two distinct market regimes in the 2008–2018 period — pre and post financial crisis recovery
- A **Random Forest** handles this non-linearity well, which contributes to the high R² score

---

## 🚀 Possible Improvements

- Add more macroeconomic features (inflation rate, interest rates, VIX)
- Use time-series-aware cross-validation (e.g. `TimeSeriesSplit`) instead of random split
- Try gradient boosting models (XGBoost, LightGBM) for comparison
- Add feature importance plot to interpret model decisions
- Deploy as a simple web app using Streamlit or Flask

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
