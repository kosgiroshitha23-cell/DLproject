"""
Spam Email Classification Using LSTM
Module: train.py
Description: Model architecture construction, model training with EarlyStopping,
saving model artifacts (Keras model, tokenizer, and config), and plotting training curves.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variables before tensorflow/matplotlib import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.matplotlib'))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, Callback

from preprocessing import (
    load_dataset,
    prepare_train_test_data,
    save_tokenizer_and_config
)


class StreamlitTrainingCallback(Callback):
    """
    Keras Callback to stream live epoch metrics to Streamlit UI or custom logger.
    """
    def __init__(self, total_epochs: int, update_fn=None):
        super().__init__()
        self.total_epochs = total_epochs
        self.update_fn = update_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)

        if self.update_fn:
            self.update_fn(epoch_num, self.total_epochs, acc, val_acc, loss, val_loss)
        else:
            print(f"Epoch [{epoch_num}/{self.total_epochs}] "
                  f"Loss: {loss:.4f} - Acc: {acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")


def build_lstm_model(vocab_size: int = 5000, embedding_dim: int = 64, max_len: int = 100, lstm_units: int = 64, dense_units: int = 32):
    """
    Constructs the Deep Learning LSTM Architecture:
    Input Text -> Text Tokenization -> Embedding Layer -> SpatialDropout1D ->
    LSTM Layer (64 units) -> Dropout Layer -> Dense Layer (32 units, ReLU) ->
    Output Layer (1 unit, Sigmoid).
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding_layer"
        ),
        SpatialDropout1D(0.2, name="spatial_dropout"),
        LSTM(
            lstm_units,
            dropout=0.2,
            name="lstm_layer"
        ),
        Dropout(0.2, name="dropout_layer"),
        Dense(
            dense_units,
            activation='relu',
            name="dense_relu_layer"
        ),
        Dense(
            1,
            activation='sigmoid',
            name="output_sigmoid_layer"
        )
    ], name="Spam_Classification_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_and_save_training_curves(history, outputs_dir: str = 'outputs'):
    """
    Generates and saves Training vs Validation Accuracy and Loss plots.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Style configuration for clean dark-modern plots
    plt.style.use('dark_background')

    # 1. Training vs Validation Accuracy Graph
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5), dpi=300)
    fig_acc.patch.set_facecolor('#0e1117')
    ax_acc.set_facecolor('#161b22')
    ax_acc.plot(epochs, history.history['accuracy'], marker='o', color='#58a6ff', linewidth=2.2, label='Training Accuracy')
    ax_acc.plot(epochs, history.history['val_accuracy'], marker='s', color='#3fb950', linewidth=2.2, linestyle='--', label='Validation Accuracy')
    ax_acc.set_title('Model Accuracy: Training vs Validation', fontsize=14, fontweight='bold', color='#f0f6fc', pad=12)
    ax_acc.set_xlabel('Epochs', fontsize=11, color='#8b949e')
    ax_acc.set_ylabel('Accuracy', fontsize=11, color='#8b949e')
    ax_acc.grid(True, linestyle=':', alpha=0.3, color='#30363d')
    ax_acc.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', fontsize=10)
    ax_acc.tick_params(colors='#8b949e')
    fig_acc.tight_layout()
    acc_path = os.path.join(outputs_dir, 'accuracy.png')
    fig_acc.savefig(acc_path)
    plt.close(fig_acc)

    # 2. Training vs Validation Loss Graph
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5), dpi=300)
    fig_loss.patch.set_facecolor('#0e1117')
    ax_loss.set_facecolor('#161b22')
    ax_loss.plot(epochs, history.history['loss'], marker='o', color='#f85149', linewidth=2.2, label='Training Loss')
    ax_loss.plot(epochs, history.history['val_loss'], marker='s', color='#d29922', linewidth=2.2, linestyle='--', label='Validation Loss')
    ax_loss.set_title('Model Loss: Training vs Validation', fontsize=14, fontweight='bold', color='#f0f6fc', pad=12)
    ax_loss.set_xlabel('Epochs', fontsize=11, color='#8b949e')
    ax_loss.set_ylabel('Binary Crossentropy Loss', fontsize=11, color='#8b949e')
    ax_loss.grid(True, linestyle=':', alpha=0.3, color='#30363d')
    ax_loss.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', fontsize=10)
    ax_loss.tick_params(colors='#8b949e')
    fig_loss.tight_layout()
    loss_path = os.path.join(outputs_dir, 'loss.png')
    fig_loss.savefig(loss_path)
    plt.close(fig_loss)

    return acc_path, loss_path


def train_lstm_pipeline(
    dataset_path: str = 'dataset/spam.csv',
    epochs: int = 10,
    batch_size: int = 32,
    max_words: int = 5000,
    max_len: int = 100,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dense_units: int = 32,
    models_dir: str = 'models',
    outputs_dir: str = 'outputs',
    progress_callback=None
):
    """
    Executes the complete model training workflow:
    1. Loads and preprocesses dataset.
    2. Tokenizes and splits data.
    3. Builds LSTM neural network.
    4. Trains with EarlyStopping.
    5. Saves model, tokenizer, and config.
    6. Generates loss and accuracy plots.
    """
    print(f"\n==========================================")
    print(f"--- Spam Email Classification LSTM Training ---")
    print(f"==========================================")
    print(f"Dataset path: {dataset_path}")
    print(f"Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, MaxWords={max_words}, MaxLen={max_len}")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Step 1: Load and preprocess dataset
    df, stats = load_dataset(dataset_path)
    print(f"Dataset Loaded Successfully: {stats['final_total']} valid records (Ham: {stats['ham_count']}, Spam: {stats['spam_count']})")

    # Step 2: Prepare train and test sets
    X_train, X_test, y_train, y_test, tokenizer, (X_train_raw, X_test_raw) = prepare_train_test_data(
        df, max_words=max_words, max_len=max_len, test_size=0.2, random_state=42
    )
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    print(f"Vocabulary Size: {vocab_size} | Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 3: Build LSTM Model
    model = build_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_len=max_len,
        lstm_units=lstm_units,
        dense_units=dense_units
    )
    model.summary()

    # Step 4: Configure Callbacks & Class Weights
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    # Calculate balanced class weights to address spam/ham imbalance
    total_samples = len(y_train)
    neg_count = float((y_train == 0).sum())
    pos_count = float((y_train == 1).sum())
    class_weights = {
        0: total_samples / (2.0 * neg_count),
        1: total_samples / (2.0 * pos_count)
    }
    print(f"Computed Class Weights: Ham(0)={class_weights[0]:.3f}, Spam(1)={class_weights[1]:.3f}")

    callbacks_list = [early_stop]
    if progress_callback:
        callbacks_list.append(StreamlitTrainingCallback(total_epochs=epochs, update_fn=progress_callback))
    else:
        callbacks_list.append(StreamlitTrainingCallback(total_epochs=epochs))

    # Step 5: Train Model
    print("\nStarting Training with EarlyStopping...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    # Step 6: Save Model, Tokenizer, and Hyperparameters
    model_save_path = os.path.join(models_dir, 'lstm_model.keras')
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    config_dict = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'max_len': max_len,
        'lstm_units': lstm_units,
        'dense_units': dense_units,
        'batch_size': batch_size,
        'epochs_requested': epochs,
        'epochs_completed': len(history.history['loss']),
        'dataset_stats': stats
    }
    tok_path, cfg_path = save_tokenizer_and_config(tokenizer, config_dict, models_dir=models_dir)
    print(f"Tokenizer saved to: {tok_path}")
    print(f"Configuration saved to: {cfg_path}")

    # Step 7: Plot and save training history
    acc_path, loss_path = plot_and_save_training_curves(history, outputs_dir=outputs_dir)
    print(f"Accuracy curve saved to: {acc_path}")
    print(f"Loss curve saved to: {loss_path}")

    # Save training history as JSON for interactive dashboards
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open(os.path.join(outputs_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)

    return model, tokenizer, config_dict, history


if __name__ == "__main__":
    dataset_file = os.path.join(BASE_DIR, 'dataset', 'spam.csv')
    train_lstm_pipeline(dataset_path=dataset_file)
