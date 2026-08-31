Nigerian Pidgin Sentiment Analysis Framework

A production-ready NLP repository fine-tuning Afro-XLM-R (`Davlan/afro-xlmr-base`) for Nigerian Pidgin (`pcm`) sentiment classification. This project benchmarked classical machine learning models against a two-stage transformer fine-tuning pipeline, resolving severe class imbalance and crowd-annotator label noise via custom class weighting, increased weight decay, lower learning rates, and label smoothing.

```text
nigerian-pidgin-sentiment
├── Dockerfile                      # Production container image configuration
├── README.md                       # Complete project documentation
├── app/                            # FastAPI REST service implementation
│   ├── main.py                     # API endpoints (/predict, /health)
│   └── schemas.py                  # Pydantic request & response data models
├── data/
│   ├── processed/                  # Cleaned & split train/val/test CSV datasets
│   └── raw/                        # Original raw dataset (pcm_raw.csv)
├── models/
│   ├── afro-xlmr-baseline/         # Stage 1 baseline model checkpoint
│   ├── afro-xlmr-weighted/         # Stage 2 weighted & regularized model (Final)
│   ├── logistic_regression_baseline.joblib
│   └── tfidf_vectorizer.joblib
├── notebooks/                      # Exploratory Data Analysis & experimental work
│   ├── 01_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_model_finetuning.ipynb
│   ├── 04_model_2_finetuning.ipynb
│   ├── 05_LR_baseline_and_model_1_comparison.ipynb
│   └── 06_models_1_and_2_comparison.ipynb
├── reports/
│   └── figures/                    # Generated evaluation charts & confusion matrices
│       └── baseline_confusion_matrix.png
├── requirements.txt                # System and Python package dependencies
├── src/                            # Modular pipeline execution scripts
│   ├── preprocessing.py            # Text cleaning and data loader functions
│   ├── baseline_training.py        # TF-IDF + Logistic Regression baseline
│   ├── model_1_training.py         # Stage 1 unweighted transformer fine-tuning
│   ├── model_2_training.py         # Stage 2 weighted & regularized fine-tuning
│   ├── evaluate_LR_baseline_and_model_1.py
│   ├── evaluate_models_1_and_2.py
│   └── predict.py                  # CLI inference module
└── tests/                          # Automated unit and API integration tests
    ├── test_api.py
    └── test_preprocessing.py
```

---

## 🏷️ Dataset & Label Schema

The pipeline maps raw sentiment text into explicit numerical class indices. Consistency is enforced across dataset processing, custom loss weighting, model config, and evaluation metrics.

Sentiment Class | Numerical ID | Target Index
| --- | --- | --- |
| **Positive** | `0` | `0` |
| **Neutral** | `1` | `1` |
| **Negative** | `2` | `2` |

```text
label2id = {"Positive": 0, "Neutral": 1, "Negative": 2}
id2label = {0: "Positive", 1: "Neutral", 2: "Negative"}
```

---

## 🔬 Iterative Methodology & Experimentation

To overcome dataset class imbalance and high crowd-annotation label noise, the modeling process followed a progressive strategy:

### 1. Classical Baseline

* **Model**: TF-IDF Vectorizer + Logistic Regression (`logistic_regression_baseline.joblib`).

* **Purpose**: Established initial speed, memory, and performance benchmarks (`src/baseline_training.py`).


### 2. Stage 1: Unweighted Fine-Tuning (afro-xlmr-baseline)

* **Model**: `Davlan/afro-xlmr-base` fine-tuned via `src/model_1_training.py`.

* **Hyperparameters**: Standard learning rate ($2\times 10^{-5}$), weight decay ($0.01$), unweighted `CrossEntropyLoss`.

* **Findings**: Overfitted to noisy target annotations and exhibited weak recall on minority classes.

### 3. Stage 2: Weighted & Regularized Fine-Tuning (afro-xlmr-weighted - Final)

* **Model**: `Davlan/afro-xlmr-base` fine-tuned via `src/model_2_training.py`.

* **Improvements**: 
    #### 1. Dynamic Class Weighting (compute_class_weight): Applied balanced class weights within a custom loss computation loop to penalize minority errors heavily:
    $$\text{Weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}$$
        
    #### 2. Decreased Learning Rate ($1\times 10^{-5}$): Reduced learning rate to prevent the optimizer from making drastic updates based on noisy ground truth labels.
    
    #### 3. Increased Weight Decay ($0.1$): Strengthened $L_2$ regularization to improve generalization.
    
    #### 4. Label Smoothing ($0.15$): Softened binary target distributions to prevent target overconfidence on mislabeled entries


### Experimentation Matrix

Feature / Parameter | Classical Baseline | Model 1 (Baseline) | Model 2 (Final Weighted)
| --- | --- | --- | --- |
| **Architecture** | TF-IDF + Logistic Reg. |afro-xlmr-base | afro-xlmr-base
| **Loss Function** | Log-Loss | Standard Cross-Entropy | Balanced Weighted Cross-Entropy
| **Learning Rate** | `N/A` | `2×10−5` | `1×10−5 (Decreased)`
| **Weight Decay** | L2​ default | 0.01 | 0.1 (Increased)
| Label Smoothing | N/A | 0.00 | 0.15 
| Checkpoint Path | models/logistic_... | models/afro-xlmr-baseline | models/afro-xlmr-weighted

---

## 🚀 Installation & Local Setup

### Prerequisite Setup

* Python 3.10+ / Linux or WSL Environment

* PyTorch with CUDA acceleration support

```text
# Clone the repository
git clone https://github.com/CodeEmmanuel-beep/Nigerian-Pidgin-Sentiment-Analyzer.git
cd nigerian-pidgin-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Running Pipelines & Training

### Data Preprocessing

Generate processed train, validation, and test splits from raw Pidgin data:

```text
Bash
python src/preprocessing.py
```

### Model Training Execution

```text
Bash
# 1. Train Classical Baseline
python src/baseline_training.py

# 2. Train Model 1 (Unweighted Baseline)
python src/model_1_training.py

# 3. Train Model 2 (Weighted & Regularized Final Model)
python src/model_2_training.py
```

### Model Evaluation

Run quantitative comparisons across saved checkpoints:

```text
Bash
# Evaluate LR vs. Model 1
python src/evaluate_LR_baseline_and_model_1.py

# Evaluate Model 1 vs. Model 2 (Final)
python src/evaluate_models_1_and_2.py
```

## 💻 API Deployment & Docker Containerization

The project includes a production-ready FastAPI application for serving sentiment predictions.

### Running FastAPI Service Locally

```text
Bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* **API Documentation**: Interactive Swagger UI is available at http://localhost:8000/docs.

### Docker Deployment

Build and run the containerized service:

```text
Bash
# Build Docker image
docker build -t pidgin-sentiment-api .

# Run container
docker run -d -p 8000:8000 --name pidgin-sentiment pidgin-sentiment-api
```

## 🧪 Testing

Automated testing covers preprocessing functions and REST API endpoints:

```text
Bash
# Execute pytest suite
pytest
```

**Project Type**: Capstone project for AI/ML course


**Student Name**: Emmanuel Eke


**Student Email**: emmanuelchiedueke01@gmail.com
