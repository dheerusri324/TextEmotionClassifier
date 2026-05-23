# Text Emotion Classifier

A deep-learning project that classifies short text into one of six emotions:
**joy · sadness · anger · fear · love · surprise**

Built with a **Bidirectional LSTM** on top of trainable word embeddings,
trained on the [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

---

## Results

| Metric | Score |
|---|---|
| Architecture | Bidirectional LSTM |
| Vocabulary size | 20,000 tokens |
| Training samples (after oversampling) | ~32,000 |
| Validation accuracy | ~90–93% |
| Evaluation | Precision / Recall / F1 per class |

> Detailed per-class metrics are saved to `saved_model/evaluation_report.txt`
> after running `evaluate.py`.

---

## Project Structure

```
textemotion/
│
├── train.py            # Train the model and save all artefacts
├── evaluate.py         # Evaluate the saved model on test.txt
├── predict.py          # Interactive / one-shot emotion prediction
│
├── tests/
│   └── test_classifier.py   # Unit tests (no GPU needed)
│
├── train.txt           # Training data  (~16,000 samples)
├── val.txt             # Validation data (~2,000 samples)
├── test.txt            # Test data       (~2,000 samples)
│
├── requirements.txt    # Python dependencies
│
└── saved_model/        # Created automatically by train.py
    ├── best_model.keras
    ├── tokenizer.pkl
    ├── label_encoder.pkl
    ├── config.json
    ├── training_curves.png
    ├── confusion_matrix.png
    └── evaluation_report.txt
```

---

## Dataset Format

Each data file (`train.txt`, `val.txt`, `test.txt`) is a semicolon-delimited
plain-text file with **no header row**:

```
i feel so happy today;joy
this makes me very sad;sadness
i am absolutely furious;anger
i am scared of the dark;fear
i love spending time with you;love
i cannot believe what just happened;surprise
```

**Column 1** — raw sentence text  
**Column 2** — emotion label (one of the six classes listed above)

---

## Setup

### 1. Clone / open the project

```bash
cd textemotion
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
python train.py
```

- Reads `train.txt` and `val.txt`
- Balances the dataset with `RandomOverSampler`
- Trains a Bidirectional LSTM with early stopping
- Saves the best model and all artefacts to `saved_model/`
- Writes a training log to `training.log`

### Evaluate

```bash
python evaluate.py
```

- Loads `saved_model/` and runs on `test.txt`
- Prints precision, recall, F1-score per class
- Saves `confusion_matrix.png` and `evaluation_report.txt`

### Predict

```bash
# Interactive mode — type sentences in the terminal
python predict.py

# One-shot mode
python predict.py --text "I am so happy today!"
```

Example output:

```
  Text    : I am so happy today!
  Emotion : JOY 😄  (97.3% confidence)

  Top predictions:
    joy        █████████████████████████  97.3%
    love       █                          3.1%
    surprise                              0.2%
```

### Run Tests

```bash
python -m pytest tests/ -v
```

Tests cover data loading, tokenisation, label encoding, and model output
shapes — **no trained model or GPU required**.

---

## Model Architecture

```
Input (max_len=100 tokens)
    │
    ▼
Embedding(vocab_size → 128-dim)
    │
    ▼
SpatialDropout1D(0.4)          ← drops entire embedding channels
    │
    ▼
Bidirectional LSTM(128 units)  ← reads left-to-right AND right-to-left
    │
    ▼
Dropout(0.4)
    │
    ▼
Dense(64, relu)
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(6, softmax)              ← one output per emotion class
```

**Why Bidirectional LSTM instead of Flatten?**  
A Flatten layer discards all positional/sequential information — it treats a
sentence as a bag of embeddings. A Bidirectional LSTM reads the sequence in
both directions so every token has context from the full sentence, which
significantly improves accuracy on NLP tasks.

---

## Hyperparameters

All hyperparameters live in the `CONFIG` dictionary at the top of `train.py`
— edit them there to experiment:

| Parameter | Default | Description |
|---|---|---|
| `max_words` | 20,000 | Vocabulary size cap |
| `max_len` | 100 | Sequence length (tokens) |
| `embed_dim` | 128 | Word embedding dimension |
| `lstm_units` | 128 | LSTM hidden units (per direction) |
| `dropout` | 0.4 | Dropout rate |
| `epochs` | 20 | Max training epochs |
| `batch_size` | 64 | Training batch size |

---

## Reproducibility

- Random seed is fixed (`random_seed: 42`) in `CONFIG`
- Training log is written to `training.log`
- Best model checkpoint saved automatically during training
- All artefacts (tokeniser, label encoder, config) serialised alongside the model

---

## Dependencies

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model training & inference |
| scikit-learn | Label encoding, metrics |
| imbalanced-learn | RandomOverSampler for class balance |
| pandas | Data loading |
| numpy | Numerical operations |
| matplotlib / seaborn | Plots & confusion matrix |
