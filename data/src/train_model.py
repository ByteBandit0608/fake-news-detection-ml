import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Example dataset
data = pd.read_csv("data/news_sample.csv")

X = data["text"]
y = data["label"]

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully")
