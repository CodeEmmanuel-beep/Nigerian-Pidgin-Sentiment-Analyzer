import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

TEST_DATA_PATH = "../data/processed/test.csv"
MODELS = {
    "Model 1 (Baseline)": "../models/afro-xlmr-baseline",
    "Model 2 (Weighted)": "../models/afro-xlmr-weighted",
}
LABEL_NAMES = ["Positive", "Neutral", "Negative"]


df_test = pd.read_csv(TEST_DATA_PATH)
text = df_test["tweet"].astype(str).tolist()
label = df_test["label"].values


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
            pred_class = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()
            preds.extend(pred_class)

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
        "Average Latency": f"{average_latency:.2f}",
    }
    return metrics, preds


def main():
    results = {}
    predictions = {}

    for name, path in MODELS.items():
        print(f"Evaluating {name}...")
        metrics, preds = evaluate_models(path, text, label)
        results[name] = metrics
        predictions[name] = preds

        print("\n" + "=" * 55)
        print(f"CLASSIFICATION REPORT: {name}")
        print("=" * 55)
        print(classification_report(label, preds, target_names=LABEL_NAMES, digits=4))

    comparison_df = pd.DataFrame(results).T
    print("\n" + "=" * 55)
    print("            MODEL COMPARISON BENCHMARK")
    print("=" * 55)
    print(comparison_df.to_string())

    # Plot and save confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, (name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(label, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Confusion Matrix - {name}")
        axes[idx].set_xlabel("Predicted Labels")
        axes[idx].set_ylabel("True Label")

    plt.tight_layout()
    os.makedirs("../reports/figures", exist_ok=True)
    plt.savefig("../reports/figures/models_1_and_2_confusion_matrices.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
