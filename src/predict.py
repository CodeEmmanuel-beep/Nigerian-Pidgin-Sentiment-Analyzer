import argparse
import sys
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configurable defaults
DEFAULT_MODEL_PATH = Path("models/afro-xlmr-weighted")
LABEL_MAPPING: dict[int, str] = {0: "Positive", 1: "Neutral", 2: "Negative"}


class SentimentPredictor:

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory non-existent at '{self.model_path}'. "
                "Ensure trained model artifacts are present."
            )

        # Load model and tokenizer onto device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()

    def predict_single(self, text: str) -> dict[str, Union[str, float]]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)

        class_idx = predicted_class.item()
        return {
            "text": text,
            "label": LABEL_MAPPING.get(class_idx, "Unknown"),
            "confidence": round(confidence.item(), 4),
        }

    def predict_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[dict[str, Union[str, float]]]:
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidences, predicted_classes = torch.max(probabilities, dim=-1)

            for text, pred_cls, conf in zip(
                batch_texts, predicted_classes, confidences
            ):
                idx = pred_cls.item()
                results.append(
                    {
                        "text": text,
                        "label": LABEL_MAPPING.get(idx, "Unknown"),
                        "confidence": round(conf.item(), 4),
                    }
                )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Nigerian Pidgin Sentiment Analysis Inference Tool"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single Pidgin text snippet to analyze",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to CSV file containing a 'tweet' or 'text' column for batch processing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output CSV path when using --file (default: predictions.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to saved model artifacts folder",
    )

    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Must provide either --text or --file argument.")

    predictor = SentimentPredictor(model_path=args.model_path)

    if args.text:
        result = predictor.predict_single(args.text)
        print("\n--- Prediction Result ---")
        print(f"Text:       {result['text']}")
        print(f"Label:      {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}\n")

    elif args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: File '{input_path}' not found.", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(input_path)
        text_column = "tweet" if "tweet" in df.columns else "text"

        if text_column not in df.columns:
            print(
                "Error: Input CSV must contain a 'tweet' or 'text' column.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Processing {len(df)} records from {input_path}...")
        results = predictor.predict_batch(df[text_column].tolist())

        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        print(f"Predictions saved successfully to '{args.output}'.")


if __name__ == "__main__":
    main()
