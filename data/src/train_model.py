import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Example dataset
data = {
    "text": [
        "Government announces new economic policy",
        "Shocking celebrity scandal revealed",
        "Scientists discover new species",
        "Fake news spreading misinformation online"
    ],
    "label": [1, 0, 1, 0]
}

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
