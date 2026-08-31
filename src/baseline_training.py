import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

FOLDER_PATH = "../data/processed"


def main():
    train_set = pd.read_csv(os.path.join(FOLDER_PATH, "train.csv"))
    val_set = pd.read_csv(os.path.join(FOLDER_PATH, "val.csv"))
    test_set = pd.read_csv(os.path.join(FOLDER_PATH, "test.csv"))

    X_train, y_train = (
        train_set["tweet"].astype(str),
        train_set["label"],
    )
    X_val, y_val = val_set["tweet"].astype(str), val_set["label"]
    X_test, y_test = test_set["tweet"].astype(str), test_set["label"]

    print(
        f"Loaded: Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=10000, sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=45)
    clf.fit(X_train_tfidf, y_train)

    print("Baseline model trained successfully")

    y_test_pred = clf.predict(X_test_tfidf)

    print("====================================================================")
    print("                    BASELINE MODEL TEST RESULT                      ")
    print("====================================================================")

    print(f"Overall Accuracy Score: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Macro F1 Score: {f1_score(y_test, y_test_pred, average='macro'):.4f}")

    print("\nDetailed Classification Report")
    target_names = ["0: Positive", "1: Neutral", "2: Negative"]
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Positive", "Neutral", "Negative"],
        yticklabels=["Positive", "Neutral", "Negative"],
    )
    plt.title("Baseline Confusion Matrix -- TF-IDF + Logistic Regression")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    os.makedirs("../reports/figures", exist_ok=True)
    plt.savefig("../reports/figures/baseline_confusion_matrix.png", dpi=300)
    plt.close()

    os.makedirs("../models", exist_ok=True)
    joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "../models/logistic_regression_baseline.joblib")

    print("Saved model artifacts to ../models/ folder")


if __name__ == "__main__":
    main()
