import streamlit as st
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_PATH = "withus/afro-xlmr-weighted"

LABEL_MAPPING = {
    0: "Positive",
    1: "Neutral",
    2: "Negative",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()


st.title("Nigerian Pidgin Sentiment Analyzer")

text = st.text_area(
    "Enter Nigerian Pidgin text", placeholder="e.g. this movie sweet die!"
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1,
            )

        confidence, predicted_class = torch.max(probs, dim=-1)

        label = LABEL_MAPPING[predicted_class.item()]

        st.subheader(label)
        st.write(f"Confidence: {confidence.item():.2%}")
