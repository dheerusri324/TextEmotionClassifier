"""
Text Emotion Classifier — Prediction Script
============================================
Loads the trained model and predicts emotion for any text — either
interactively (type sentences in the terminal) or as a one-off via CLI.

Prerequisites:
    Run  train.py  first so that the saved_model/ directory exists.

Usage:
    # Interactive mode
    python predict.py

    # One-shot mode
    python predict.py --text "I am so happy today!"
"""

import os
import json
import pickle
import argparse
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = "saved_model"

# Emoji for a friendlier terminal experience
EMOTION_EMOJI = {
    "joy":      "😄",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "love":     "❤️ ",
    "surprise": "😲",
}


# ══════════════════════════════════════════════
# Load artefacts
# ══════════════════════════════════════════════

def load_artifacts():
    """
    Load the trained model, tokenizer, label encoder, and config.

    All four files are created by train.py and stored in saved_model/.
    Raises a helpful FileNotFoundError if they are missing.
    """
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"'{MODEL_DIR}' directory not found.\n"
            "Please run  train.py  first to train and save the model."
        )

    model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    return model, tokenizer, label_encoder, config


# ══════════════════════════════════════════════
# Core prediction function
# ══════════════════════════════════════════════

def predict(text: str, model, tokenizer, label_encoder, config) -> tuple:
    """
    Predict the emotion of a single text string.

    Returns
    -------
    emotion   : str   — top-predicted emotion label
    confidence: dict  — {emotion_label: probability} for all classes
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded   = pad_sequences(
        sequence,
        maxlen=config["max_len"],
        padding="post",
        truncating="post",
    )
    probs    = model.predict(padded, verbose=0)[0]
    top_idx  = int(np.argmax(probs))
    emotion  = label_encoder.inverse_transform([top_idx])[0]

    confidence = {
        label_encoder.classes_[i]: float(probs[i])
        for i in range(len(label_encoder.classes_))
    }
    return emotion, confidence


def _print_result(text: str, emotion: str, confidence: dict) -> None:
    """Pretty-print a single prediction result with a bar chart."""
    emoji = EMOTION_EMOJI.get(emotion, "")
    top3  = sorted(confidence.items(), key=lambda x: -x[1])[:3]

    print(f"\n  Text    : {text}")
    print(f"  Emotion : {emotion.upper()} {emoji}  "
          f"({confidence[emotion] * 100:.1f}% confidence)\n")
    print("  Top predictions:")
    for emo, prob in top3:
        bar = "█" * int(prob * 25)
        print(f"    {emo:<10}  {bar:<25} {prob * 100:.1f}%")
    print()


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the emotion expressed in a piece of text."
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to classify. Omit to enter interactive mode.",
    )
    args = parser.parse_args()

    print("Loading model … ", end="", flush=True)
    model, tokenizer, label_encoder, config = load_artifacts()
    print("done.\n")

    if args.text:
        # ── Single prediction mode ────────────────────────────────────────
        emotion, confidence = predict(
            args.text, model, tokenizer, label_encoder, config
        )
        _print_result(args.text, emotion, confidence)

    else:
        # ── Interactive mode ──────────────────────────────────────────────
        banner = "=" * 52
        print(banner)
        print("   Text Emotion Classifier  —  Interactive Mode")
        print("   Emotions: joy | sadness | anger | fear | love | surprise")
        print("   Type  'exit'  to quit.")
        print(banner)

        while True:
            try:
                user_input = input("\n📝  Enter a sentence: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            emotion, confidence = predict(
                user_input, model, tokenizer, label_encoder, config
            )
            _print_result(user_input, emotion, confidence)


if __name__ == "__main__":
    main()
