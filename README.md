# Nigerian Pidgin Sentiment Analysis Framework

A production-ready NLP repository fine-tuning Afro-XLM-R (`Davlan/afro-xlmr-base`) for Nigerian Pidgin (`pcm`) sentiment classification. This project benchmarks classical machine learning models against a two-stage transformer fine-tuning pipeline, addressing severe class imbalance and crowd-annotator label noise through custom class weighting, increased weight decay, lower learning rates, and label smoothing.

The final fine-tuned model is hosted on Hugging Face as `withus/afro-xlmr-weighted` and deployed as an interactive Streamlit NLP application.

```text
nigerian-pidgin-sentiment

├── README.md                         # Complete project documentation

├── app/                            
    |-- main.py                       # Streamlit NLP application

├── data/
│   ├── processed/                    # Cleaned & split train/val/test CSV datasets
│   └── raw/                          # Original raw dataset (pcm_raw.csv)

├── notebooks/                        # Exploratory Data Analysis & experimental work
│   ├── 01_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_model_finetuning.ipynb
│   ├── 04_model_2_finetuning.ipynb
│   ├── 05_LR_baseline_and_model_1_comparison.ipynb
│   └── 06_models_1_and_2_comparison.ipynb

├── reports/
│   └── figures/                      # Generated evaluation charts & confusion matrices
│       └── baseline_confusion_matrix.png

├── requirements.txt                  # Python package dependencies

├── src/                              # Modular pipeline execution scripts
│   ├── preprocessing.py              # Text cleaning and data loader functions
│   ├── baseline_training.py          # TF-IDF + Logistic Regression baseline
│   ├── model_1_training.py           # Stage 1 unweighted transformer fine-tuning
│   ├── model_2_training.py           # Stage 2 weighted & regularized fine-tuning
│   ├── evaluate_LR_baseline_and_model_1.py
│   ├── evaluate_models_1_and_2.py
│   └── predict.py                    # CLI inference module

└── tests/                            # Automated tests
     |-- test_hf_model.py
     └── test_preprocessing.py         # Preprocessing tests
```

> The large fine-tuned transformer checkpoint is hosted separately on Hugging Face rather than committed to the Git repository.

---

## 🏷️ Dataset & Label Schema

The pipeline maps raw sentiment text into explicit numerical class indices. Consistency is enforced across dataset processing, custom loss weighting, model configuration, and evaluation metrics.

| Sentiment Class | Numerical ID | Target Index |
| --------------- | -----------: | -----------: |
| **Positive**    |          `0` |          `0` |
| **Neutral**     |          `1` |          `1` |
| **Negative**    |          `2` |          `2` |

```python
label2id = {"Positive": 0, "Neutral": 1, "Negative": 2}

id2label = {
    0: "Positive",
    1: "Neutral",
    2: "Negative",
}
```

---

## 🔬 Iterative Methodology & Experimentation

To address dataset class imbalance and high crowd-annotation label noise, the modeling process followed a progressive strategy.

### 1. Classical Baseline

* **Model:** TF-IDF Vectorizer + Logistic Regression (`logistic_regression_baseline.joblib`)
* **Purpose:** Established initial speed, memory, and performance benchmarks (`src/baseline_training.py`).

### 2. Stage 1: Unweighted Fine-Tuning (`afro-xlmr-baseline`)

* **Base Model:** `Davlan/afro-xlmr-base`
* **Training:** Fine-tuned via `src/model_1_training.py`.
* **Hyperparameters:** Learning rate `2×10⁻⁵`, weight decay `0.01`, and standard unweighted `CrossEntropyLoss`.
* **Findings:** The model overfit noisy target annotations and exhibited weak recall on minority classes.

### 3. Stage 2: Weighted & Regularized Fine-Tuning (`afro-xlmr-weighted` — Final)

* **Base Model:** `Davlan/afro-xlmr-base`
* **Training:** Fine-tuned via `src/model_2_training.py`.
* **Final Model:** `withus/afro-xlmr-weighted`

#### Improvements

1. **Dynamic Class Weighting:** Applied balanced class weights within a custom loss computation loop to penalize minority-class errors more heavily.

   \(\text{Weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}\)

2. **Decreased Learning Rate:** Reduced the learning rate to `1×10⁻⁵` to prevent drastic optimizer updates caused by noisy ground-truth labels.

3. **Increased Weight Decay:** Increased weight decay from `0.01` to `0.1` to strengthen L2 regularization and improve generalization.

4. **Label Smoothing:** Applied `0.15` label smoothing to reduce target overconfidence on potentially mislabeled examples.

### Experimentation Matrix

| Feature / Parameter | Classical Baseline                    | Model 1                | Model 2 — Final                 |
| ------------------- | ------------------------------------- | ---------------------- | ------------------------------- |
| **Architecture**    | TF-IDF + Logistic Regression          | Afro-XLM-R             | Afro-XLM-R                      |
| **Loss Function**   | Log-Loss                              | Standard Cross-Entropy | Balanced Weighted Cross-Entropy |
| **Learning Rate**   | `N/A`                                 | `2×10⁻⁵`               | `1×10⁻⁵`                        |
| **Weight Decay**    | L2 default                            | `0.01`                 | `0.1`                           |
| **Label Smoothing** | `N/A`                                 | `0.00`                 | `0.15`                          |
| **Checkpoint**      | `logistic_regression_baseline.joblib` | `afro-xlmr-baseline`   | `withus/afro-xlmr-weighted`     |

---

## 🚀 Installation & Local Setup

### Prerequisites

* Python 3.10+
* PyTorch
* Git

Clone the repository:

```bash
git clone https://github.com/CodeEmmanuel-beep/Nigerian-Pidgin-Sentiment-Analyzer.git

cd Nigerian-Pidgin-Sentiment-Analyzer
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Running Pipelines & Training

### Data Preprocessing

Generate processed train, validation, and test splits from the raw Pidgin dataset:

```bash
python src/preprocessing.py
```

### Model Training

```bash
# 1. Train Classical Baseline
python src/baseline_training.py

# 2. Train Model 1 — Unweighted Baseline
python src/model_1_training.py

# 3. Train Model 2 — Weighted & Regularized Final Model
python src/model_2_training.py
```

### Model Evaluation

```bash
# Evaluate Logistic Regression vs. Model 1
python src/evaluate_LR_baseline_and_model_1.py

# Evaluate Model 1 vs. Model 2
python src/evaluate_models_1_and_2.py
```

---

## 🌐 Streamlit Deployment

The final NLP model is deployed as an interactive Streamlit application.

The application code is hosted in the GitHub repository, while the large fine-tuned transformer checkpoint is hosted on Hugging Face.

```text
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
Hugging Face Hub
        ↓
withus/afro-xlmr-weighted
        ↓
Nigerian Pidgin Sentiment Prediction
```

The Streamlit application automatically installs the dependencies specified in `requirements.txt` and loads the fine-tuned model from Hugging Face at runtime.

The model is loaded once using Streamlit's resource caching to avoid unnecessary model reloading during application interaction.

---

## 🧪 Testing

Automated testing covers the preprocessing pipeline.

Run the test suite with:

```bash
python -m pytest
```

---

## 📌 Project Information

**Project Type:** Capstone project for AI/ML course

**Student Name:** Emmanuel Eke

**Email:** emmanuelchiedueke01@gmail.com

**Student Email:** [emmanuelchiedueke01@gmail.com](mailto:emmanuelchiedueke01@gmail.com)
