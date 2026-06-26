import os
import re
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """
    Clean the news article text.
    """
    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def prepare_data():
    """
    Load dataset, preprocess text, split data,
    and return TF-IDF features.
    """

    # Project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dataset paths
    fake_path = os.path.join(BASE_DIR, "data", "Fake.csv")
    true_path = os.path.join(BASE_DIR, "data", "True.csv")

    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Labels
    fake_df["label"] = 0
    true_df["label"] = 1

    # Merge
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle
    df = shuffle(df, random_state=42)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Features and labels
    X = df["clean_text"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    print("Module Loaded Successfully!")
