"""
Spam Email Classification Using LSTM
Module: preprocessing.py
Description: Natural Language Processing (NLP) text cleaning, tokenization,
sequence padding, dataset loading, and automatic column detection.
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import nltk

# Ensure cache dirs stay within project to avoid Windows/sandbox permission issues
os.environ.setdefault('KERAS_HOME', os.path.abspath(os.path.join(os.path.dirname(__file__), '.keras')))
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '.matplotlib')))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Ensure NLTK data is downloaded
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    _ = stopwords.words('english')
except Exception:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set of English stopwords
STOPWORDS = set(stopwords.words('english'))

# Common column name heuristics
LABEL_COLUMN_CANDIDATES = ['v1', 'label', 'category', 'class', 'target', 'spam', 'type', 'status']
MESSAGE_COLUMN_CANDIDATES = ['v2', 'message', 'text', 'sms', 'email', 'body', 'content', 'mail', 'msg']


def detect_columns(df: pd.DataFrame):
    """
    Automatically detects message and label columns in the given DataFrame.
    Supports standard CSV formats (e.g. v1/v2, label/message, category/text).
    """
    cols_lower = {col: str(col).strip().lower() for col in df.columns}
    label_col = None
    message_col = None

    # Step 1: Match by common column names
    for col, col_lower in cols_lower.items():
        if col_lower in LABEL_COLUMN_CANDIDATES and label_col is None:
            label_col = col
        elif col_lower in MESSAGE_COLUMN_CANDIDATES and message_col is None:
            message_col = col

    # Step 2: Fallback heuristic based on content if not matched
    if label_col is None or message_col is None:
        for col in df.columns:
            if label_col is None:
                unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
                # If column has 2-3 distinct values likely indicating binary label
                if unique_vals.issubset({'ham', 'spam', '0', '1', 'legit', 'not spam'}):
                    label_col = col
                    continue

            if message_col is None and col != label_col:
                # Text columns typically have high string length variance
                sample_lens = df[col].dropna().astype(str).apply(len)
                if sample_lens.mean() > 15:
                    message_col = col

    # Fallback to first two columns if still undetermined
    if label_col is None and len(df.columns) >= 1:
        label_col = df.columns[0]
    if message_col is None and len(df.columns) >= 2:
        message_col = df.columns[1] if df.columns[1] != label_col else df.columns[0]

    return label_col, message_col


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans raw text for NLP processing:
    1. Converts text into lowercase.
    2. Strips HTML tags and URLs.
    3. Removes unnecessary special characters and numbers.
    4. Removes extra spaces.
    5. Tokenizes and optionally filters out stopwords.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Keep only alphabetic letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Word tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    return " ".join(tokens)


def load_dataset(file_path: str):
    """
    Loads, validates, and prepares the dataset from a CSV file.
    - Automatically maps columns.
    - Encodes Ham as 0 and Spam as 1.
    - Removes missing and duplicate records.
    Returns:
        tuple: (cleaned_df, raw_stats, label_col, message_col)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Try common encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unable to read CSV file {file_path} with supported encodings.")

    raw_total = len(df)
    label_col, message_col = detect_columns(df)

    if not label_col or not message_col:
        raise ValueError(f"Could not automatically identify message and label columns in {df.columns.tolist()}")

    # Select only label and message columns
    df = df[[label_col, message_col]].copy()
    df.rename(columns={label_col: 'label_raw', message_col: 'message_raw'}, inplace=True)

    # Drop missing values
    df.dropna(subset=['label_raw', 'message_raw'], inplace=True)
    after_na = len(df)

    # Normalize labels: Ham -> 0, Spam -> 1
    def map_label(val):
        val_str = str(val).strip().lower()
        if val_str in ['ham', '0', '0.0', 'legit', 'not spam']:
            return 0
        elif val_str in ['spam', '1', '1.0']:
            return 1
        return None

    df['label'] = df['label_raw'].apply(map_label)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # Drop duplicate messages
    duplicates_count = df.duplicated(subset=['message_raw']).sum()
    df.drop_duplicates(subset=['message_raw'], inplace=True)
    cleaned_total = len(df)

    # Apply text cleaning
    df['clean_message'] = df['message_raw'].apply(clean_text)

    # Filter out empty cleaned messages
    df = df[df['clean_message'].str.strip().str.len() > 0].copy()

    stats = {
        'raw_total': raw_total,
        'missing_dropped': raw_total - after_na,
        'duplicates_dropped': int(duplicates_count),
        'final_total': len(df),
        'ham_count': int((df['label'] == 0).sum()),
        'spam_count': int((df['label'] == 1).sum()),
        'label_col_detected': label_col,
        'message_col_detected': message_col
    }

    return df, stats


def fit_tokenizer_on_data(texts, max_words: int = 5000, oov_token: str = "<OOV>"):
    """
    Fits Keras Tokenizer exclusively on training text to prevent data leakage.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token, lower=True)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def transform_texts_to_padded_sequences(tokenizer, texts, max_len: int = 100):
    """
    Converts list of texts to padded integer sequences.
    Uses padding='pre' so that words are positioned at the final recurrent steps of the LSTM.
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='pre', truncating='pre')
    return padded


def prepare_train_test_data(df: pd.DataFrame, max_words: int = 5000, max_len: int = 100, test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into train and test sets and creates tokenized padded arrays.
    Prevents data leakage by fitting tokenizer only on X_train.
    """
    X = df['clean_message'].values
    y = df['label'].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit tokenizer on training texts only
    tokenizer = fit_tokenizer_on_data(X_train_raw, max_words=max_words)

    # Transform both train and test
    X_train = transform_texts_to_padded_sequences(tokenizer, X_train_raw, max_len=max_len)
    X_test = transform_texts_to_padded_sequences(tokenizer, X_test_raw, max_len=max_len)

    return X_train, X_test, y_train, y_test, tokenizer, (X_train_raw, X_test_raw)


def save_tokenizer_and_config(tokenizer, config_dict: dict, models_dir: str = 'models'):
    """
    Saves tokenizer to tokenizer.pkl and hyperparameters to config.json.
    """
    os.makedirs(models_dir, exist_ok=True)
    tok_path = os.path.join(models_dir, 'tokenizer.pkl')
    cfg_path = os.path.join(models_dir, 'config.json')

    with open(tok_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4)

    return tok_path, cfg_path


def load_tokenizer_and_config(models_dir: str = 'models'):
    """
    Loads saved tokenizer and configuration dictionary.
    """
    tok_path = os.path.join(models_dir, 'tokenizer.pkl')
    cfg_path = os.path.join(models_dir, 'config.json')

    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tok_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found at: {cfg_path}")

    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return tokenizer, config
