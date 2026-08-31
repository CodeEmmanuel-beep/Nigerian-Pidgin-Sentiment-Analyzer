import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)

MODEL_CKPT = "Davlan/afro-xlmr-base"
FOLDER_PATH = "../data/processed"
OUTPUT_DIR = "../models/afro-xlmr-baseline"

ID2LABEL = {0: "Positive", 1: "Neutral", 2: "Negative"}
LABEL2ID = {"Positive": 0, "Neutral": 1, "Negative": 2}


def tokenize_batch(example, tokenizer):
    return tokenizer(example["tweet"], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    print(f"PyTorch version: {torch.__version__}")
    device_available = torch.cuda.is_available()
    print(f"CUDA Available: {device_available}")
    if device_available:
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    print("\nLoading datasets...")
    train_df = pd.read_csv(os.path.join(FOLDER_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(FOLDER_PATH, "val.csv"))

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print(f"Loading tokenizer: {MODEL_CKPT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, token=HF_TOKEN)

    tokenized_train = train_dataset.map(
        lambda x: tokenize_batch(x, tokenizer), batched=True
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_batch(x, tokenizer), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        token=HF_TOKEN,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=45,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting baseline model fine-tuning...")
    trainer.train()

    print(f"\nSaving baseline model checkpoint to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Baseline training complete!")


if __name__ == "__main__":
    main()
