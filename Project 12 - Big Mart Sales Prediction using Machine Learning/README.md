# 🛒 Big Mart Sales Prediction

A machine learning project to predict item sales across Big Mart outlets using XGBoost regression. The model is trained on historical sales data and uses various item and outlet features to generate predictions.

---

## 📌 Project Overview

Big Mart collects sales data for different products across various stores. The goal of this project is to build a predictive model that estimates the sales (`Item_Outlet_Sales`) for each product at a given store. This helps the retail chain understand which products and properties drive higher sales.

---

## 📂 Dataset

- **Source:** [Kaggle – BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?select=Train.csv)
- **Training samples:** 8,523 rows
- **Features:** 11 input columns + 1 target column

| Column | Description |
|---|---|
| `Item_Identifier` | Unique product ID |
| `Item_Weight` | Weight of the product |
| `Item_Fat_Content` | Whether the product is low fat or not |
| `Item_Visibility` | Percentage of display area in store |
| `Item_Type` | Category of the product |
| `Item_MRP` | Maximum retail price |
| `Outlet_Identifier` | Unique store ID |
| `Outlet_Establishment_Year` | Year the store was established |
| `Outlet_Size` | Size of the store (Small/Medium/High) |
| `Outlet_Location_Type` | Type of city |
| `Outlet_Type` | Whether grocery store or supermarket |
| `Item_Outlet_Sales` | **Target variable** – sales of the product |

---

## 🔧 Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mirzasameer2000/Machine-Learning-Projects.git
cd Machine-Learning-Projects/BigMart-Sales-Prediction
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### 3. Download the Dataset

Download `Train.csv` from [Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?select=Train.csv) and place it in the project directory.

### 4. Run the Notebook

Open `Big_Mart_Sales_Prediction_using_Machine_Learning.ipynb` in Google Colab or Jupyter Notebook and run all cells.

---

## 🧪 Workflow

### 1. Data Loading & Exploration
- Loaded dataset using Pandas
- Checked shape, data types, and null values
- Found 1,463 missing values in `Item_Weight` and 2,410 in `Outlet_Size`

### 2. Data Preprocessing
- Filled missing `Item_Weight` with column mean
- Filled missing `Outlet_Size` using mode per `Outlet_Type` (pivot table approach)
- Standardized `Item_Fat_Content` labels (`LF` → `Low Fat`, `reg` → `Regular`, `low fat` → `Low Fat`)
- Applied `LabelEncoder` to all categorical columns

### 3. Exploratory Data Analysis
- Distribution plots for `Item_Weight`, `Item_Visibility`, `Item_MRP`, `Item_Outlet_Sales`
- Count plots for `Outlet_Establishment_Year` and `Item_Type`
- Observed that `Item_Outlet_Sales` is right-skewed

### 4. Model Training
- Split data: 80% train / 20% test (`random_state=2`)
- Trained `XGBRegressor` with default parameters

### 5. Evaluation
- Evaluated on both train and test sets using R² score and MAE

---

## 📊 Results

| Dataset | R² Score |
|---|---|
| Training | ~0.87 |
| Testing | ~0.50 |

> Note: Results may slightly vary based on XGBoost version and random state.

---

## 📁 Project Structure

```
BigMart-Sales-Prediction/
│
├── Big_Mart_Sales_Prediction_using_Machine_Learning.ipynb   # Main notebook
├── Train.csv                        # Dataset (download from Kaggle)
└── README.md
```

---

## 🙋 Author

**Muhammad Sameer**
- 📧 Email: [sameermubasher99@gmail.com](mailto:sameermubasher99@gmail.com)
- 🌐 Portfolio: [www.muhammadsameer.de](https://www.muhammadsameer.de)
- 💼 LinkedIn: [linkedin.com/in/mirzasameer](https://www.linkedin.com/in/mirzasameer)
- 🐙 GitHub: [github.com/mirzasameer2000](https://github.com/mirzasameer2000)
- 🔗 Xing: [xing.com/profile/MuhammadSameer](https://www.xing.com/profile/MuhammadSameer)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
