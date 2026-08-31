import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

#
TEST_DATA_PATH = "../data/processed/test.csv"
TFIDF_VECTORIZER_PATH = "../models/tfidf_vectorizer.joblib"
LOGISTIC_REGRESSION_PATH = "../models/logistic_regression_baseline.joblib"
AFRO_XLMR_PATH = "../models/afro-xlmr-best/afro-xlmr-baseline"
LABEL_NAMES = ["Positive", "Neutral", "Negative"]

df_test = pd.read_csv(TEST_DATA_PATH)
text = df_test["tweet"].astype(str).tolist()
label = df_test["label"].values


def evaluate_saved_joblib_model(tfidf_path, baseline_path, text_data, label_data):
    print(f"\nLoading saved Traditional Baseline from: {baseline_path}")

    vectorizer = joblib.load(tfidf_path)
    model = joblib.load(baseline_path)

    X_test = vectorizer.transform(text_data)

    start_time = time.time()
    preds = model.predict(X_test)
    total_time = time.time() - start_time
    average_latency = (total_time / len(text_data)) * 1000

    precision, recall, f1, _ = precision_recall_fscore_support(
        label_data, preds, average="macro"
    )
    acc = accuracy_score(label_data, preds)

    metrics = {
        "Accuracy": f"{acc:.4f}",
        "Macro F1": f"{f1:.4f}",
        "Macro Precision": f"{precision:.4f}",
        "Macro Recall": f"{recall:.4f}",
        "Average Latency": f"{average_latency:.4f}",
    }

    return metrics, preds


def evaluate_models(model_path, text_data, label_data):
    print(f"\nEvaluating model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    start_time = time.time()
    preds = []
    batch_size = 50

    with torch.no_grad():
        for i in range(0, len(text_data), batch_size):
            chunk = text_data[i : i + batch_size]
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()
            preds.extend(batch_preds)

    total_time = time.time() - start_time
    average_latency = (total_time / len(text_data)) * 1000
    print(f"Average Latency: {average_latency:.2f} ms")

    precision, recall, f1, _ = precision_recall_fscore_support(
        label_data, preds, average="macro"
    )
    acc = accuracy_score(label_data, preds)

    metrics = {
        "Accuracy": f"{acc:.4f}",
        "Macro F1": f"{f1:.4f}",
        "Macro Precision": f"{precision:.4f}",
        "Macro Recall": f"{recall:.4f}",
        "Average Latency": f"{average_latency:.2}",
    }

    return metrics, preds


def main():
    results = {}
    predictions = {}

    # Evaluate Traditional Baseline
    name = "Traditional Baseline"
    print(f"Evaluating {name}...")
    metrics, preds = evaluate_saved_joblib_model(
        TFIDF_VECTORIZER_PATH, LOGISTIC_REGRESSION_PATH, text, label
    )
    results[name] = metrics
    predictions[name] = preds

    print("\n" + "=" * 55)
    print(f"CLASSIFICATION REPORT: {name}")
    print("=" * 55)
    print(classification_report(label, preds, target_names=LABEL_NAMES, digits=4))

    # Evaluate Afro-XLMR Best Model
    name = "AFRO_XLMR"
    print(f"Evaluating {name}...")
    metrics, preds = evaluate_models(AFRO_XLMR_PATH, text, label)
    results[name] = metrics
    predictions[name] = preds

    print("\n" + "=" * 55)
    print(f"CLASSIFICATION REPORT: {name}")
    print("=" * 55)
    print(classification_report(label, preds, target_names=LABEL_NAMES, digits=4))

    # Print Benchmark Summary Table
    comparison_df = pd.DataFrame(results).T
    print("\n" + "=" * 55)
    print("       TRADITIONAL BASELINE VS AFRO-XLMR BEST")
    print("=" * 55)
    print(comparison_df.to_string())

    # Plot & Save Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (model_name, model_preds) in enumerate(predictions.items()):
        cm = confusion_matrix(label, model_preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Confusion Matrix - {model_name}")
        axes[idx].set_xlabel("Predicted Labels")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    os.makedirs("../reports/figures", exist_ok=True)
    plt.savefig(
        "../reports/figures/traditional_vs_afro_xlmr_confusion_matrices.png",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
