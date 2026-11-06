"""
Spam Email Classification Using LSTM
Module: prediction.py
Description: Inference engine that takes raw email or SMS text, applies the NLP
cleaning pipeline, pads the sequence, runs inference through the trained LSTM model,
computes confidence scores, and provides explainability breakdown.
"""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.matplotlib'))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from preprocessing import (
    clean_text,
    transform_texts_to_padded_sequences,
    load_tokenizer_and_config
)

# Common indicators and phishing / spam trigger lexicon
SPAM_TRIGGER_KEYWORDS = {
    'free', 'win', 'winner', 'won', 'cash', 'prize', 'claim', 'urgent', 'urgent!',
    'call', 'txt', 'reply', 'text', 'selected', 'award', 'awarded', 'guaranteed',
    'credit', 'loan', 'bank', 'account', 'verify', 'password', 'expire', 'limited',
    'congratulations', 'congrats', 'offer', 'exclusive', 'bonus', 'gift', 'voucher',
    'subscribe', 'ringtone', 'customer', 'service', 'action', 'alert', 'draw'
}


class SpamPredictor:
    """
    Spam prediction handler that caches loaded LSTM model and tokenizer.
    """
    _instance = None

    def __init__(self, models_dir: str = 'models'):
        self.models_dir = os.path.abspath(models_dir)
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_artifacts()

    def load_artifacts(self):
        model_path = os.path.join(self.models_dir, 'lstm_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Please train the model first by running `python train.py`."
            )
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer, self.config = load_tokenizer_and_config(self.models_dir)

    def predict(self, raw_text: str) -> dict:
        """
        Performs inference on a single raw text string.
        Returns a comprehensive dictionary with prediction, confidence, and explainability.
        """
        if not raw_text or not str(raw_text).strip():
            raise ValueError("Input message cannot be empty or purely whitespace.")

        # Step 1: Clean raw input text
        cleaned = clean_text(str(raw_text))
        if not cleaned:
            # Fallback if text contained only special symbols/stopwords
            cleaned = str(raw_text).lower().strip()

        # Step 2: Sequence transformation using fitted tokenizer
        max_len = self.config.get('max_len', 100)
        padded_seq = transform_texts_to_padded_sequences(self.tokenizer, [cleaned], max_len=max_len)

        # Step 3: Run inference through the LSTM model
        raw_prob = float(self.model.predict(padded_seq, verbose=0)[0][0])

        # Step 4: Classification and confidence calculations
        is_spam = raw_prob >= 0.5
        prediction_label = "SPAM" if is_spam else "HAM"
        confidence = (raw_prob if is_spam else (1.0 - raw_prob)) * 100.0

        # Step 5: Explainability & suspicious keyword extraction
        words_in_text = set(cleaned.split())
        flagged_keywords = sorted(list(words_in_text.intersection(SPAM_TRIGGER_KEYWORDS)))

        # Unknown tokens (out of vocabulary)
        word_index = self.tokenizer.word_index
        tokens = cleaned.split()
        oov_words = [w for w in tokens if w not in word_index]

        # Generate human-readable explanation
        if is_spam:
            if flagged_keywords:
                reasons = f"Detected high-risk promotional / urgency terms: {', '.join([f"'{w}'" for w in flagged_keywords])}."
            else:
                reasons = "The sequential syntax pattern matches typical unsolicited promotional or phishing structures learned by the LSTM."
            explanation = (
                f"This message was classified as **SPAM** with **{confidence:.1f}% confidence**. "
                f"{reasons} Caution is advised before responding or clicking any links."
            )
        else:
            explanation = (
                f"This message was classified as **NOT SPAM (HAM)** with **{confidence:.1f}% confidence**. "
                f"Its vocabulary and sequential patterns resemble standard interpersonal or legitimate correspondence."
            )

        return {
            'prediction': prediction_label,
            'is_spam': is_spam,
            'spam_probability': raw_prob,
            'confidence': confidence,
            'raw_text': raw_text,
            'cleaned_text': cleaned,
            'tokens_count': len(tokens),
            'detected_keywords': flagged_keywords,
            'oov_words_count': len(oov_words),
            'explanation': explanation
        }


# Singleton helper function for easy reuse
_predictor_cache = None

def predict_message(text: str, models_dir: str = 'models') -> dict:
    global _predictor_cache
    if _predictor_cache is None or _predictor_cache.models_dir != os.path.abspath(models_dir):
        _predictor_cache = SpamPredictor(models_dir=models_dir)
    return _predictor_cache.predict(text)


if __name__ == "__main__":
    import sys
    models_directory = os.path.join(BASE_DIR, 'models')
    sample = (
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Draw! "
        "Txt the word CLAIM to 81010 to receive your prize."
    )
    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])

    try:
        res = predict_message(sample, models_dir=models_directory)
        print("\n--- Inference Result ---")
        print(f"Input Text: {sample}")
        print(f"Prediction: {res['prediction']} (Confidence: {res['confidence']:.2f}%)")
        print(f"Spam Probability: {res['spam_probability']:.4f}")
        print(f"Explanation: {res['explanation']}")
    except Exception as e:
        print("Prediction Error:", e)
