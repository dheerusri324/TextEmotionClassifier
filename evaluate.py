import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # silence TensorFlow C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"  # silence oneDNN messages

"""
Text Emotion Classifier — Evaluation Script
============================================
Loads the saved model and evaluates it on the held-out test set (test.txt).

Outputs
-------
• Classification report (precision / recall / F1 per class + macro avg)
• Confusion matrix heatmap  →  saved_model/confusion_matrix.png
• Plain-text report         →  saved_model/evaluation_report.txt

Usage:
    python evaluate.py
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger(__name__)

MODEL_DIR = "saved_model"
TEST_FILE  = "test.txt"


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def load_artifacts():
    """
    Load the trained model and all preprocessing artefacts from saved_model/.

    Raises FileNotFoundError if train.py has not been run yet.
    """
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"'{MODEL_DIR}' directory not found. "
            "Please run  train.py  first to create the saved model."
        )
    log.info("Loading saved artefacts...")
    model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    log.info("  Artefacts loaded successfully.")
    return model, tokenizer, label_encoder, config


def load_data(filepath: str) -> pd.DataFrame:
    """Load a semicolon-delimited text;emotion file, dropping invalid rows."""
    log.info(f"Loading test data from: {filepath}")
    df = pd.read_csv(filepath, sep=";", header=None, names=["Text", "Emotions"])
    df.dropna(inplace=True)
    df = df[df["Text"].str.strip() != ""]
    log.info(f"  {len(df)} test samples")
    return df


def plot_confusion_matrix(cm, class_names, save_path: str) -> None:
    """Plot a labelled confusion-matrix heatmap and save to disk."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix — Test Set", fontsize=14, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def main() -> None:
    # ── 1. Load model & artefacts ─────────────────────────────────────────
    model, tokenizer, label_encoder, config = load_artifacts()

    # ── 2. Load & preprocess test set ────────────────────────────────────
    test_df = load_data(TEST_FILE)
    sequences = tokenizer.texts_to_sequences(test_df["Text"].tolist())
    X_test = pad_sequences(
        sequences,
        maxlen=config["max_len"],
        padding="post",
        truncating="post",
    )
    y_true = label_encoder.transform(test_df["Emotions"])

    # ── 3. Predict ────────────────────────────────────────────────────────
    log.info("\nRunning predictions on test set...")
    y_pred_probs = model.predict(X_test, batch_size=config["batch_size"], verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # ── 4. Classification report ──────────────────────────────────────────
    class_names = label_encoder.classes_
    report = classification_report(y_true, y_pred, target_names=class_names)

    sep = "=" * 55
    full_report = (
        f"Text Emotion Classifier — Test-Set Evaluation\n"
        f"{sep}\n\n"
        f"{report}\n"
        f"{sep}\n"
        f"Test accuracy : {np.mean(y_pred == y_true) * 100:.2f}%\n"
        f"Test samples  : {len(y_true)}\n"
    )

    log.info(f"\n{full_report}")

    # Save to file
    report_path = os.path.join(MODEL_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(full_report)
    log.info(f"Evaluation report saved → {report_path}")

    # ── 5. Confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm, class_names,
        save_path=os.path.join(MODEL_DIR, "confusion_matrix.png"),
    )

    log.info(f"\n✅  Evaluation complete. Results saved to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
