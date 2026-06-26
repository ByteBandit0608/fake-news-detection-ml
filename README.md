# Fake News Detection using Machine Learning

This project builds a machine learning model to classify news articles as real or fake using text analysis.

## Technologies
- Python
- Pandas
- Scikit-learn

## Project Structure
data/ – dataset files  
src/ – model training code  

## How to Run
pip install -r requirements.txt
python src/train_model.py
notebooks/ – exploratory analysis and model experimentation

# 📰 Fake News Detection using Machine Learning

A machine learning web application that predicts whether a news article is **Real** or **Fake** using **Logistic Regression** and **TF-IDF Vectorization**.

Built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🚀 Features

- Detects whether a news article is Real or Fake
- Interactive web interface using Streamlit
- Prediction confidence score
- Probability distribution for both classes
- Live word and character count
- Clean and responsive UI
- Trained on nearly **45,000** news articles

---

## 🖥️ Application Preview

### Home Page

![Home](assets/home.png)

---

### Fake News Prediction

![Fake Prediction](assets/fake_prediction.png)

---

### Real News Prediction

![Real Prediction](assets/real_prediction.png)

---

## 🧠 Machine Learning Pipeline

1. Load the Fake.csv and True.csv datasets
2. Clean and preprocess the news text
3. Convert text into TF-IDF feature vectors
4. Train a Logistic Regression classifier
5. Save the trained model and vectorizer using Joblib
6. Deploy the model using Streamlit

---

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **98.94%** |
| Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |
| Dataset Size | 44,898 Articles |

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit
- Matplotlib
- WordCloud

---

## 📁 Project Structure

```text
fake-news-detection-ml/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_comparison.ipynb
│
├── results/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── assets/
│   ├── home.png
│   ├── fake_prediction.png
│   └── real_prediction.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ByteBandit0608/fake-news-detection-ml.git
```

Move into the project directory

```bash
cd fake-news-detection-ml
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## 🔮 Future Improvements

- Support for transformer-based models (BERT)
- Explainable AI using SHAP/LIME
- News source credibility analysis
- URL-based fake news detection
- Multi-language support

---

## 👨‍💻 Author

**Bhavya Posham**

AI & Machine Learning Undergraduate (2028)

Interested in Machine Learning, Computer Vision, NLP, and AI applications.