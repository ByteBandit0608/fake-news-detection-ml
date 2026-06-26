import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import prepare_data


def train_model():

    # Load processed data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data()

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save models
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "models")

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "logistic_regression.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

    print("\nModel saved successfully!")
    print("Vectorizer saved successfully!")


if __name__ == "__main__":
    train_model()