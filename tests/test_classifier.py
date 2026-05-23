"""
Text Emotion Classifier — Unit Tests
======================================
Tests core logic without requiring a trained model (fast, no GPU needed).

Covers:
  • Data loading (valid file, missing file, malformed rows)
  • Text preprocessing (tokenising, padding, OOV handling)
  • Prediction output shape and type contracts
  • Label encoding / decoding round-trip

Run with:
    python -m pytest tests/ -v
  or
    python tests/test_classifier.py
"""

import os
import sys
import json
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

# ── Make the project root importable ─────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Import helpers from train.py
from train import load_data, build_model, CONFIG


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_DATA = (
    "i feel so happy today;joy\n"
    "this makes me very sad;sadness\n"
    "i am absolutely furious;anger\n"
    "i am scared of the dark;fear\n"
    "i love spending time with you;love\n"
    "i cannot believe what just happened;surprise\n"
)


def make_temp_file(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Test Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoading(unittest.TestCase):
    """Tests for the load_data() helper in train.py."""

    def test_valid_file_returns_dataframe(self):
        path = make_temp_file(SAMPLE_DATA)
        try:
            df = load_data(path)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn("Text", df.columns)
            self.assertIn("Emotions", df.columns)
        finally:
            os.unlink(path)

    def test_correct_row_count(self):
        path = make_temp_file(SAMPLE_DATA)
        try:
            df = load_data(path)
            self.assertEqual(len(df), 6)
        finally:
            os.unlink(path)

    def test_blank_rows_are_dropped(self):
        # Add a blank line and an all-whitespace text row
        data_with_blanks = SAMPLE_DATA + "\n   ;joy\n"
        path = make_temp_file(data_with_blanks)
        try:
            df = load_data(path)
            self.assertEqual(len(df), 6)           # blank rows removed
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(Exception):
            load_data("/nonexistent/path/file.txt")

    def test_all_six_emotions_present(self):
        path = make_temp_file(SAMPLE_DATA)
        try:
            df = load_data(path)
            self.assertEqual(df["Emotions"].nunique(), 6)
        finally:
            os.unlink(path)


class TestPreprocessing(unittest.TestCase):
    """Tests for text tokenisation and sequence padding."""

    def setUp(self):
        self.texts = [
            "i feel so happy today",
            "this makes me very sad",
            "i cannot believe what just happened",
        ]
        self.tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.texts)
        self.max_len = 10

    def test_sequences_have_correct_length(self):
        seqs = self.tokenizer.texts_to_sequences(self.texts)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding="post")
        self.assertEqual(padded.shape[1], self.max_len)

    def test_oov_token_handles_unknown_words(self):
        unknown_text = ["zxqwerty blarg flurp"]
        seq = self.tokenizer.texts_to_sequences(unknown_text)
        # OOV index is 1 by default when oov_token is set
        self.assertIn(1, seq[0])

    def test_empty_string_produces_padded_zeros(self):
        seq = self.tokenizer.texts_to_sequences([""])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post")
        self.assertTrue(np.all(padded == 0))

    def test_long_text_is_truncated(self):
        long_text = ["word " * 50]
        seq = self.tokenizer.texts_to_sequences(long_text)
        padded = pad_sequences(seq, maxlen=self.max_len, truncating="post")
        self.assertEqual(padded.shape[1], self.max_len)


class TestLabelEncoding(unittest.TestCase):
    """Tests for label encoding / decoding round-trips."""

    def setUp(self):
        self.emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
        self.le = LabelEncoder()
        self.le.fit(self.emotions)

    def test_all_classes_encoded(self):
        self.assertEqual(len(self.le.classes_), 6)

    def test_encode_decode_roundtrip(self):
        for emo in self.emotions:
            encoded = self.le.transform([emo])
            decoded = self.le.inverse_transform(encoded)[0]
            self.assertEqual(emo, decoded)

    def test_encoded_values_are_integers(self):
        encoded = self.le.transform(self.emotions)
        for val in encoded:
            self.assertIsInstance(int(val), int)

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            self.le.transform(["disgust"])


class TestModelArchitecture(unittest.TestCase):
    """Sanity-checks for the Bidirectional LSTM model."""

    def setUp(self):
        self.cfg = CONFIG.copy()
        self.cfg["max_len"]    = 20
        self.cfg["embed_dim"]  = 32
        self.cfg["lstm_units"] = 32
        self.cfg["dropout"]    = 0.2
        self.vocab_size  = 500
        self.num_classes = 6

    def test_model_builds_without_error(self):
        model = build_model(self.vocab_size, self.num_classes, self.cfg)
        self.assertIsNotNone(model)

    def test_output_shape_matches_num_classes(self):
        model = build_model(self.vocab_size, self.num_classes, self.cfg)
        # Shape: (batch, num_classes)
        self.assertEqual(model.output_shape, (None, self.num_classes))

    def test_prediction_probabilities_sum_to_one(self):
        model = build_model(self.vocab_size, self.num_classes, self.cfg)
        dummy_input = np.zeros((1, self.cfg["max_len"]), dtype=np.int32)
        probs = model.predict(dummy_input, verbose=0)[0]
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_prediction_output_is_valid_probability(self):
        model = build_model(self.vocab_size, self.num_classes, self.cfg)
        dummy_input = np.zeros((3, self.cfg["max_len"]), dtype=np.int32)
        probs = model.predict(dummy_input, verbose=0)
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
