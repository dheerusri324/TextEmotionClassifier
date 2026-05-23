# Text Emotion Classifier

A deep-learning project that classifies short text into one of six emotions:
**joy · sadness · anger · fear · love · surprise**

Built with a **Bidirectional LSTM** and **GloVe pre-trained embeddings**,
trained on the [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

---

## Results

| Metric | Score |
|---|---|
| Architecture | Bidirectional LSTM |
| Embeddings | GloVe 6B 100d (fine-tuned) |
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
├── train.py              # Train the model and save all artefacts
├── evaluate.py           # Evaluate the saved model on test.txt
├── predict.py            # Interactive / one-shot emotion prediction
├── download_glove.py     # Download GloVe pre-trained embeddings (~350 MB)
│
├── tests/
│   └── test_classifier.py    # Unit tests (no GPU needed)
│
├── train.txt             # Training data  (~16,000 samples)
├── val.txt               # Validation data (~2,000 samples)
├── test.txt              # Test data       (~2,000 samples)
│
├── requirements.txt      # Python dependencies
│
├── glove/                # Created by download_glove.py (git-ignored)
│   └── glove.6B.100d.txt
│
└── saved_model/          # Created automatically by train.py (git-ignored)
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

### 1. Clone the repository

```bash
git clone https://github.com/dheerusri324/TextEmotionClassifier
cd TextEmotionClassifier
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

### 4. Download GloVe embeddings (recommended)

GloVe gives every word a pre-trained semantic meaning learned from billions of
words, so the model handles words it has never seen in the training data.

```bash
python download_glove.py
```

> Downloads `glove.6B.100d.txt` (~350 MB) into `glove/`.  
> Skip this step if you want to train with random embeddings instead — `train.py`
> will detect the missing file and fall back automatically.

---

## Usage

### Train

```bash
python train.py
```

- Reads `train.txt` (training) and `val.txt` (validation)
- Balances the dataset with `RandomOverSampler`
- Loads GloVe embeddings if `glove/glove.6B.100d.txt` exists
- Trains a Bidirectional LSTM with EarlyStopping + ReduceLROnPlateau
- Saves the best checkpoint and all artefacts to `saved_model/`
- Writes a full training log to `training.log`

### Evaluate

```bash
python evaluate.py
```

- Loads `saved_model/` and runs on the held-out `test.txt`
- Prints precision, recall, F1-score per emotion class
- Saves `confusion_matrix.png` and `evaluation_report.txt` to `saved_model/`

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
  Emotion : JOY 😄  (99.9% confidence)

  Top predictions:
    joy        █████████████████████████  99.9%
    sadness                               0.1%
    fear                                  0.1%
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
Embedding(vocab_size → 100-dim)    ← GloVe pre-trained, fine-tuned during training
    │
    ▼
SpatialDropout1D(0.4)              ← drops entire embedding channels
    │
    ▼
Bidirectional LSTM(128 units)      ← reads left-to-right AND right-to-left
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
Dense(6, softmax)                  ← one output per emotion class
```

**Why Bidirectional LSTM instead of Flatten?**  
A Flatten layer discards all word-order information — it treats a sentence as an
unordered bag of embeddings. A Bidirectional LSTM reads the sequence in both
directions so every token has context from the full sentence.

**Why GloVe embeddings?**  
Without GloVe, words not seen in the training data (e.g. "trophy", "medal") get
random, meaningless vectors. With GloVe, those words already carry semantic
meaning learned from billions of words — "trophy" already knows it's related to
"win", "award", "achievement" — so the model predicts correctly even for
out-of-vocabulary inputs.

---

## Hyperparameters

All hyperparameters live in the `CONFIG` dictionary at the top of `train.py`
— edit them there to experiment:

| Parameter | Default | Description |
|---|---|---|
| `max_words` | 20,000 | Vocabulary size cap |
| `max_len` | 100 | Sequence length (tokens) |
| `embed_dim` | 100 | Embedding dimension (must match GloVe file) |
| `glove_path` | `glove/glove.6B.100d.txt` | Path to GloVe file, or `None` to skip |
| `lstm_units` | 128 | LSTM hidden units (per direction) |
| `dropout` | 0.4 | Dropout rate |
| `epochs` | 20 | Max training epochs (EarlyStopping may stop sooner) |
| `batch_size` | 64 | Training batch size |

---

## Reproducibility

- Random seed is fixed (`random_seed: 42`) in `CONFIG`
- Training log is written to `training.log`
- Best model checkpoint saved automatically via `ModelCheckpoint`
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
