"""
Spam Email Classification Using LSTM
Module: evaluation.py
Description: Evaluates the trained LSTM model on the test dataset. Computes accuracy,
precision, recall, F1-score, confusion matrix, and generates high-resolution visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.matplotlib'))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import (
    load_dataset,
    prepare_train_test_data,
    load_tokenizer_and_config
)


def evaluate_model_pipeline(
    dataset_path: str = 'dataset/spam.csv',
    models_dir: str = 'models',
    outputs_dir: str = 'outputs'
):
    """
    Evaluates the trained LSTM model on held-out test data and saves performance charts.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'lstm_model.keras')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

    # Load model, tokenizer and configuration
    model = tf.keras.models.load_model(model_path)
    tokenizer, config = load_tokenizer_and_config(models_dir)

    # Load and process dataset
    df, stats = load_dataset(dataset_path)
    max_words = config.get('vocab_size', 5000)
    max_len = config.get('max_len', 100)

    # Prepare train and test split with same random seed
    X_train, X_test, y_train, y_test, _, (X_train_raw, X_test_raw) = prepare_train_test_data(
        df, max_words=max_words, max_len=max_len, test_size=0.2, random_state=42
    )

    # Run predictions on test set
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Calculate Core Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    report_dict = classification_report(
        y_test, y_pred, target_names=['HAM (Legit)', 'SPAM'], output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred, target_names=['HAM (Legit)', 'SPAM']
    )

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report_dict,
        'test_samples_count': len(y_test),
        'test_ham_count': int((y_test == 0).sum()),
        'test_spam_count': int((y_test == 1).sum())
    }

    # Save metrics JSON
    metrics_path = os.path.join(outputs_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    # Generate Visualizations
    plt.style.use('dark_background')

    # 1. Confusion Matrix Heatmap
    cm_arr = np.array(cm)
    fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5), dpi=300)
    fig_cm.patch.set_facecolor('#0e1117')
    ax_cm.set_facecolor('#161b22')

    group_counts = [f"{v:,}" for v in cm_arr.flatten()]
    group_percentages = [f"{v:.2%}" for v in cm_arr.flatten() / np.sum(cm_arr)]
    group_names = ['True Neg (Ham)', 'False Pos (Spam)', 'False Neg (Ham)', 'True Pos (Spam)']
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm_arr,
        annot=labels,
        fmt="",
        cmap='Blues',
        cbar=True,
        ax=ax_cm,
        xticklabels=['Pred HAM', 'Pred SPAM'],
        yticklabels=['Actual HAM', 'Actual SPAM'],
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax_cm.set_title('Test Set Confusion Matrix', fontsize=14, fontweight='bold', color='#f0f6fc', pad=12)
    ax_cm.set_xlabel('Predicted Label', fontsize=11, color='#8b949e')
    ax_cm.set_ylabel('True Label', fontsize=11, color='#8b949e')
    ax_cm.tick_params(colors='#8b949e')
    fig_cm.tight_layout()
    cm_path = os.path.join(outputs_dir, 'confusion_matrix.png')
    fig_cm.savefig(cm_path)
    plt.close(fig_cm)

    # 2. Performance Comparison Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5), dpi=300)
    fig_bar.patch.set_facecolor('#0e1117')
    ax_bar.set_facecolor('#161b22')
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [acc * 100, prec * 100, rec * 100, f1 * 100]
    bar_colors = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff']

    bars = ax_bar.bar(metric_names, metric_values, color=bar_colors, width=0.55, edgecolor='#30363d', linewidth=1.2)
    ax_bar.set_ylim(0, 105)
    ax_bar.set_ylabel('Percentage (%)', fontsize=11, color='#8b949e')
    ax_bar.set_title('LSTM Classification Performance Metrics', fontsize=14, fontweight='bold', color='#f0f6fc', pad=12)
    ax_bar.grid(axis='y', linestyle=':', alpha=0.3, color='#30363d')
    ax_bar.tick_params(colors='#8b949e')

    for bar, val in zip(bars, metric_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.2f}%",
            ha='center',
            va='bottom',
            fontsize=10.5,
            fontweight='bold',
            color='#f0f6fc'
        )
    fig_bar.tight_layout()
    comp_path = os.path.join(outputs_dir, 'metrics_comparison.png')
    fig_bar.savefig(comp_path)
    plt.close(fig_bar)

    # 3. Spam vs Ham Distribution Chart
    fig_dist, ax_dist = plt.subplots(figsize=(7, 4.5), dpi=300)
    fig_dist.patch.set_facecolor('#0e1117')
    ax_dist.set_facecolor('#161b22')
    classes = ['HAM (Legitimate)', 'SPAM']
    counts = [stats['ham_count'], stats['spam_count']]
    colors = ['#2ea043', '#f85149']

    dist_bars = ax_dist.bar(classes, counts, color=colors, width=0.45, edgecolor='#30363d', linewidth=1.2)
    ax_dist.set_ylabel('Message Count', fontsize=11, color='#8b949e')
    ax_dist.set_title('SMS Spam Collection Class Distribution', fontsize=14, fontweight='bold', color='#f0f6fc', pad=12)
    ax_dist.grid(axis='y', linestyle=':', alpha=0.3, color='#30363d')
    ax_dist.tick_params(colors='#8b949e')

    total_msgs = sum(counts)
    for bar, count in zip(dist_bars, counts):
        pct = (count / total_msgs) * 100
        ax_dist.text(
            bar.get_x() + bar.get_width() / 2,
            count + (max(counts) * 0.02),
            f"{count:,} ({pct:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=10.5,
            fontweight='bold',
            color='#f0f6fc'
        )
    fig_dist.tight_layout()
    dist_path = os.path.join(outputs_dir, 'distribution.png')
    fig_dist.savefig(dist_path)
    plt.close(fig_dist)

    print("\n==========================================")
    print("--- Model Evaluation on Test Dataset ---")
    print("==========================================")
    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"Test Precision : {prec * 100:.2f}%")
    print(f"Test Recall    : {rec * 100:.2f}%")
    print(f"Test F1-Score  : {f1 * 100:.2f}%")
    print("\nClassification Report:\n", report_text)
    print(f"Metrics saved to: {metrics_path}")
    print(f"Visualizations saved to: {outputs_dir}")

    return metrics, {
        'confusion_matrix': cm_path,
        'metrics_comparison': comp_path,
        'distribution': dist_path
    }


if __name__ == "__main__":
    dataset_file = os.path.join(BASE_DIR, 'dataset', 'spam.csv')
    evaluate_model_pipeline(dataset_path=dataset_file)
