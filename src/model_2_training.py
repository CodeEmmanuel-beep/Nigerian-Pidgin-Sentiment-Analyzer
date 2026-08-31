import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)

MODEL_CKPT = "Davlan/afro-xlmr-base"
FOLDER_PATH = "../data/processed"
OUTPUT_DIR = "../models/afro-xlmr-weighted"

ID2LABEL = {0: "Positive", 1: "Neutral", 2: "Negative"}
LABEL2ID = {"Positive": 0, "Neutral": 1, "Negative": 2}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


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
    device_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA Available: {device_available}")
    if device_available:
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    print("\nLoading datasets...")
    train_df = pd.read_csv(os.path.join(FOLDER_PATH, "train.csv"))
    val_df = pd.read_csv(os.path.join(FOLDER_PATH, "val.csv"))

    pos_df = train_df[train_df["label"] == 1]

    train_df = pd.concat([train_df, pos_df, pos_df], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=45).reset_index(drop=True)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label"],
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float)

    print(f"Computed Class Weights: {weight_tensor}")

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

    num_train_epochs = 6
    per_device_train_batch_size = 16
    warmup_ratio = 0.1
    total_steps = (
        int(len(tokenized_train) / per_device_train_batch_size) * num_train_epochs
    )
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.1,
        label_smoothing_factor=0.15,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=45,
        fp16=device_available,
    )

    trainer = WeightedTrainer(
        class_weights=weight_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting weighted model fine-tuning...")
    trainer.train()

    print(f"\nSaving weighted model checkpoint to '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Weighted model training complete!")


if __name__ == "__main__":
    main()
