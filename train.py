import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # silence TensorFlow C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"  # silence oneDNN messages

"""
Text Emotion Classifier — Training Script
==========================================
Trains a Bidirectional LSTM model to classify text into one of six emotions:
  joy, sadness, anger, fear, love, surprise

GloVe Pre-trained Embeddings (optional but recommended)
--------------------------------------------------------
GloVe gives every word a rich, pre-trained meaning vector learned from
billions of words. This means even words the model has never seen in the
training data (e.g. "trophy", "medal") already have semantic meaning —
they know they're related to "win", "award", "achievement" — so the model
can make much better predictions on unseen vocabulary.

To enable GloVe:
    python download_glove.py      ← downloads glove.6B.100d.txt (~350 MB)
    python train.py               ← will auto-detect and use GloVe

Without GloVe:
    python train.py               ← falls back to random embeddings (still works)

Usage:
    python train.py
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# ─────────────────────────────────────────────
# Hyperparameters & paths — edit here to experiment
# ─────────────────────────────────────────────
CONFIG = {
    # Data
    "train_file":  "train.txt",
    "val_file":    "val.txt",
    "model_dir":   "saved_model",
    # Tokeniser
    "max_words":   20000,   # Vocabulary cap (most-frequent N words kept)
    "max_len":     100,     # Sequences padded/truncated to this many tokens
    # Model
    "embed_dim":   100,     # Must match GloVe dimension (50 / 100 / 200 / 300)
    "lstm_units":  128,     # LSTM hidden units per direction
    "dropout":     0.4,
    # GloVe — set to None to skip and use random embeddings instead
    "glove_path":  "glove/glove.6B.100d.txt",
    # Training
    "epochs":      20,
    "batch_size":  64,
    "random_seed": 42,
}

# ─────────────────────────────────────────────
# Logging — writes to console AND training.log
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a semicolon-delimited  text;emotion  file.
    Returns a DataFrame with columns ['Text', 'Emotions'].
    Rows that are empty or contain NaN are dropped.
    """
    log.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, sep=";", header=None, names=["Text", "Emotions"])
    before = len(df)
    df.dropna(inplace=True)
    df = df[df["Text"].str.strip() != ""]
    after = len(df)
    if before != after:
        log.warning(f"  Dropped {before - after} invalid rows from {filepath}.")
    log.info(f"  {after} samples | {df['Emotions'].nunique()} unique emotions")
    return df


def explore_data(df: pd.DataFrame, label: str = "Dataset") -> None:
    """Log key statistics for a dataset split."""
    sep = "─" * 45
    log.info(f"\n{sep}\n  {label} statistics\n{sep}")
    log.info(f"  Samples        : {len(df)}")
    log.info(f"  Unique emotions: {df['Emotions'].nunique()}")
    dist = df["Emotions"].value_counts().to_string()
    log.info(f"  Class distribution:\n{dist}")
    lens = df["Text"].str.split().str.len()
    log.info(
        f"  Text length (words) — mean: {lens.mean():.1f} "
        f"| max: {lens.max()} | min: {lens.min()}"
    )


# ══════════════════════════════════════════════
# GloVe embeddings
# ══════════════════════════════════════════════

def load_glove_embeddings(
    glove_path: str, word_index: dict, embed_dim: int, max_words: int
) -> np.ndarray:
    """
    Load GloVe pre-trained word vectors and build an embedding matrix
    aligned with our tokenizer's word index.

    Why GloVe?
    ──────────
    Without GloVe, the Embedding layer starts with random numbers and only
    learns from words that appear in the training data. A word like "trophy"
    that never appeared in train.txt gets a random, meaningless vector.

    With GloVe, "trophy" already starts with a vector that captures its
    relationship to "win", "award", "achievement", "prize" — learned from
    billions of words across Wikipedia and news articles. The model can
    immediately reason about it correctly.

    Words not found in GloVe fall back to small random initialisation so
    training can still fine-tune them from context.
    """
    log.info(f"Loading GloVe embeddings from: {glove_path}")
    glove_vectors = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word   = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_vectors[word] = vector

    log.info(f"  Loaded {len(glove_vectors):,} GloVe vectors (dim={embed_dim})")

    vocab_size = min(len(word_index) + 1, max_words + 1)

    # Random small init for words not in GloVe (better than zeros — they can
    # still be learned during fine-tuning instead of being invisible)
    embedding_matrix = np.random.normal(
        scale=0.1, size=(vocab_size, embed_dim)
    ).astype(np.float32)
    embedding_matrix[0] = 0  # index 0 = padding → always zero vector

    found = 0
    for word, idx in word_index.items():
        if idx >= max_words:
            continue
        if word in glove_vectors:
            embedding_matrix[idx] = glove_vectors[word]
            found += 1

    total = min(len(word_index), max_words)
    pct   = found / total * 100
    log.info(f"  GloVe coverage : {found:,} / {total:,} vocabulary words ({pct:.1f}%)")
    if pct < 50:
        log.warning("  Low GloVe coverage — check that embed_dim matches the GloVe file.")
    return embedding_matrix


# ══════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════

def build_model(
    vocab_size: int,
    num_classes: int,
    cfg: dict,
    embedding_matrix: np.ndarray = None,
) -> Sequential:
    """
    Build a Bidirectional LSTM emotion classifier.

    Architecture rationale
    ─────────────────────
    • Embedding          – Maps token IDs to dense vectors.
                           If embedding_matrix is provided, initialised from GloVe
                           (trainable=True so they are fine-tuned during training).
    • SpatialDropout1D   – Drops entire embedding channels; more effective than
                           regular dropout for sequential / embedding data.
    • Bidirectional LSTM – Reads the sequence left-to-right AND right-to-left,
                           giving each token context from the full sentence.
    • Dropout + Dense    – Standard dense head for classification.
    """
    if embedding_matrix is not None:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=cfg["embed_dim"],
            input_length=cfg["max_len"],
            weights=[embedding_matrix],
            trainable=True,   # fine-tune GloVe vectors during training
        )
        log.info("  Embedding layer : GloVe pre-trained (fine-tuning enabled)")
    else:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=cfg["embed_dim"],
            input_length=cfg["max_len"],
        )
        log.info("  Embedding layer : randomly initialised")

    model = Sequential([
        embedding_layer,
        SpatialDropout1D(cfg["dropout"]),
        Bidirectional(LSTM(cfg["lstm_units"], return_sequences=False)),
        Dropout(cfg["dropout"]),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════

def plot_history(history, save_dir: str) -> None:
    """Save accuracy and loss curves to saved_model/training_curves.png."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Training curves saved → {path}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main() -> None:
    np.random.seed(CONFIG["random_seed"])
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

    # ── 1. Load & explore ────────────────────────────────────────────────
    train_df = load_data(CONFIG["train_file"])
    val_df   = load_data(CONFIG["val_file"])
    explore_data(train_df, "Training Set")
    explore_data(val_df,   "Validation Set")

    # ── 2. Tokenise ──────────────────────────────────────────────────────
    log.info("\nTokenising text...")
    tokenizer = Tokenizer(num_words=CONFIG["max_words"], oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["Text"].tolist())   # fit on TRAIN only
    vocab_size = min(len(tokenizer.word_index) + 1, CONFIG["max_words"] + 1)
    log.info(f"  Vocabulary size : {vocab_size:,}")

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            seqs, maxlen=CONFIG["max_len"], padding="post", truncating="post"
        )

    X_train = encode(train_df["Text"].tolist())
    X_val   = encode(val_df["Text"].tolist())

    # ── 3. Encode labels ─────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Emotions"])
    y_train_int = label_encoder.transform(train_df["Emotions"])
    y_val_int   = label_encoder.transform(val_df["Emotions"])
    num_classes = len(label_encoder.classes_)
    log.info(f"  Emotion classes ({num_classes}): {list(label_encoder.classes_)}")

    # ── 4. Balance training set ───────────────────────────────────────────
    log.info("\nBalancing training set with RandomOverSampler...")
    ros = RandomOverSampler(random_state=CONFIG["random_seed"])
    X_train, y_train_int = ros.fit_resample(X_train, y_train_int)
    log.info(f"  Training samples after resampling: {len(X_train):,}")

    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_val   = to_categorical(y_val_int,   num_classes=num_classes)

    # ── 5. Load GloVe (if available) ─────────────────────────────────────
    embedding_matrix = None
    glove_path = CONFIG.get("glove_path")

    if glove_path and os.path.isfile(glove_path):
        embedding_matrix = load_glove_embeddings(
            glove_path, tokenizer.word_index, CONFIG["embed_dim"], CONFIG["max_words"]
        )
    elif glove_path:
        log.warning(f"\nGloVe file not found at '{glove_path}'.")
        log.warning("Run  python download_glove.py  to download it (~350 MB).")
        log.warning("Training will proceed with random embeddings.\n")

    # ── 6. Build model ────────────────────────────────────────────────────
    log.info("\nBuilding model...")
    model = build_model(vocab_size, num_classes, CONFIG, embedding_matrix)
    model.summary(print_fn=log.info)

    # ── 7. Callbacks ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(CONFIG["model_dir"], "best_model.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-5, verbose=1,
        ),
    ]

    # ── 8. Train ──────────────────────────────────────────────────────────
    using_glove = "with GloVe" if embedding_matrix is not None else "without GloVe (random embeddings)"
    log.info(f"\nTraining {using_glove} — up to {CONFIG['epochs']} epochs "
             f"(batch_size={CONFIG['batch_size']})...\n")
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # ── 9. Save artefacts ─────────────────────────────────────────────────
    tok_path = os.path.join(CONFIG["model_dir"], "tokenizer.pkl")
    le_path  = os.path.join(CONFIG["model_dir"], "label_encoder.pkl")
    cfg_path = os.path.join(CONFIG["model_dir"], "config.json")

    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(cfg_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    log.info(f"\nSaved → {ckpt_path}")
    log.info(f"Saved → {tok_path}")
    log.info(f"Saved → {le_path}")
    log.info(f"Saved → {cfg_path}")

    # ── 10. Validation classification report ─────────────────────────────
    log.info("\nValidation set — classification report:")
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(
        y_val_int, y_val_pred, target_names=label_encoder.classes_
    )
    log.info(f"\n{report}")

    # ── 11. Save plots ────────────────────────────────────────────────────
    plot_history(history, CONFIG["model_dir"])

    log.info("\n✅  Training complete!")
    log.info("   Run  evaluate.py  for full test-set metrics.")
    log.info("   Run  predict.py   for interactive emotion prediction.")


if __name__ == "__main__":
    main()
