import torch
from app.schemas import SentimentRequest, SentimentResponse
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(title="Nigerian Pidgin Sentiment Analyzer API", version="1.0.0")

MODEL_PATH = "withus/afro-xlmr-weighted"
LABEL_MAPPING = {0: "Positive", 1: "Neutral", 2: "Negative"}

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts from {MODEL_PATH}: {e}")


@app.get("/")
def health_check():
    return {"status": "ok", "model_in_use": "afro-xlmr-weighted"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(payload: SentimentRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)

    idx = predicted_class.item()
    return SentimentResponse(
        text=payload.text,
        label=LABEL_MAPPING.get(idx, "Unknown"),
        confidence=round(confidence.item(), 4),
    )
