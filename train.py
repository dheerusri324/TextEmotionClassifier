"""
Text Emotion Classifier — Training Script
==========================================
Trains a Bidirectional LSTM model to classify text into one of six emotions:
  joy, sadness, anger, fear, love, surprise

What this script does:
  1. Loads and explores train.txt / val.txt
  2. Tokenises text and pads sequences to a fixed length
  3. Applies RandomOverSampler to balance class counts
  4. Builds a Bidirectional LSTM model (better than Flatten for NLP)
  5. Trains with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
  6. Saves model + tokenizer + label-encoder + config to saved_model/
  7. Prints a classification report on the validation set

Usage:
    python train.py
"""

import os
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
    "embed_dim":   128,     # Size of each word embedding vector
    "lstm_units":  128,     # Number of LSTM hidden units per direction
    "dropout":     0.4,     # Dropout rate after embedding & after LSTM
    # Training
    "epochs":      20,      # Maximum epochs (EarlyStopping may stop it sooner)
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
    df = df[df["Text"].str.strip() != ""]          # drop blank-text rows
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
# Model
# ══════════════════════════════════════════════

def build_model(vocab_size: int, num_classes: int, cfg: dict) -> Sequential:
    """
    Build a Bidirectional LSTM classifier.

    Architecture rationale
    ─────────────────────
    • Embedding          – Converts token IDs to dense vectors; trainable from scratch.
    • SpatialDropout1D   – Drops entire embedding channels; more effective than
                           regular dropout for sequential data.
    • Bidirectional LSTM – Reads the sequence left-to-right AND right-to-left,
                           so each token has context from the full sentence.
                           Significantly outperforms a simple Flatten approach.
    • Dropout            – Standard regularisation before the dense head.
    • Dense(64, relu)    – Intermediate representation.
    • Dense(softmax)     – One output neuron per emotion class.
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=cfg["embed_dim"],
            input_length=cfg["max_len"],
        ),
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
    log.info(f"  Vocabulary size: {vocab_size:,}")

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(
            seqs,
            maxlen=CONFIG["max_len"],
            padding="post",
            truncating="post",
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

    # ── 5. Build model ────────────────────────────────────────────────────
    log.info("\nBuilding model...")
    model = build_model(vocab_size, num_classes, CONFIG)
    model.summary(print_fn=log.info)

    # ── 6. Callbacks ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(CONFIG["model_dir"], "best_model.keras")
    callbacks = [
        # Stop early if val_accuracy doesn't improve for 4 epochs
        EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
        # Save the best checkpoint
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        # Halve the learning rate when val_loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-5, verbose=1,
        ),
    ]

    # ── 7. Train ──────────────────────────────────────────────────────────
    log.info(f"\nTraining — up to {CONFIG['epochs']} epochs "
             f"(batch_size={CONFIG['batch_size']})...\n")
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # ── 8. Save artefacts ─────────────────────────────────────────────────
    # best_model.keras is already saved by ModelCheckpoint; save the rest:
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

    # ── 9. Validation classification report ──────────────────────────────
    log.info("\nValidation set — classification report:")
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(
        y_val_int, y_val_pred, target_names=label_encoder.classes_
    )
    log.info(f"\n{report}")

    # ── 10. Save plots ────────────────────────────────────────────────────
    plot_history(history, CONFIG["model_dir"])

    log.info("\n✅  Training complete!")
    log.info("   Run  evaluate.py  for full test-set metrics.")
    log.info("   Run  predict.py   for interactive emotion prediction.")


if __name__ == "__main__":
    main()
