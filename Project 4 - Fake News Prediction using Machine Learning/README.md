# 📰 Fake News Prediction using Machine Learning

## 📌 Project Overview
This project focuses on detecting whether a news article is **real or fake** using Machine Learning techniques.  
A **Logistic Regression model** is trained on a labeled dataset of news articles to classify them accurately.

---

## 📊 Dataset
The dataset used in this project is obtained from Kaggle:

https://www.kaggle.com/datasets/jainpooja/fake-news-detection/data

It consists of two CSV files:
- True.csv → Real news articles  
- Fake.csv → Fake news articles  

### Dataset Features:
- title → News headline  
- text → Full article content  
- subject → Category of news  
- date → Publication date  
- label → Target variable  
  - 0 → Real News  
  - 1 → Fake News  

---

## ⚙️ Project Workflow

### 1. Data Loading
- Imported datasets using Pandas  
- Loaded both real and fake news datasets  

### 2. Data Labeling
- Assigned labels:
  - Real news → 0
  - Fake news → 1

### 3. Data Merging & Shuffling
- Combined both datasets into one  
- Shuffled data to ensure randomness  
- Reset index and added unique id column  

### 4. Feature Engineering
- Merged title and text into a single column  

### 5. Text Preprocessing
- Converted text to lowercase  
- Removed special characters  
- Removed stopwords (using NLTK)  
- Applied stemming using PorterStemmer  

### 6. Text Vectorization
- Used TF-IDF Vectorizer  

### 7. Model Training
- Split dataset into training and testing sets  
- Used Logistic Regression  

### 8. Model Evaluation
- Training Accuracy: 99.17%  
- Testing Accuracy: 98.76%  

### 9. Prediction System
- Predicts whether news is Real or Fake  

---

## 🧠 Technologies Used
- Python  
- Pandas  
- NumPy  
- NLTK  
- Scikit-learn  

---

## 📁 Project Structure
Fake-News-Prediction/
│
├── True.csv
├── Fake.csv
├── news_dataset.csv
├── Fake News Prediction using Machine Learning.ipynb
├── README.md

---

## 🚀 How to Run

1. Install dependencies:
pip install numpy pandas scikit-learn nltk

2. Run the notebook in Jupyter / VS Code / Google Colab

---

## 📌 Key Insights
- Text preprocessing significantly improves accuracy  
- Logistic Regression performs well for NLP tasks  
- TF-IDF is effective for feature extraction  

---

## 🔮 Future Improvements
- Try advanced models (Random Forest, XGBoost, BERT)  
- Build web app using Flask/Django  
- Deploy as API  

---

## 👨‍💻 Author
Muhammad Sameer  

---

## ⭐ Acknowledgment
Dataset provided by Kaggle:
https://www.kaggle.com/datasets/jainpooja/fake-news-detection/data
