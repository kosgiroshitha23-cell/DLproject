# 🛡️ Spam Email Classification Using LSTM (Deep Learning)

An end-to-end Natural Language Processing (NLP) and Deep Learning system designed for intelligent email and SMS spam detection using a **Long Short-Term Memory (LSTM)** neural network. Built with **TensorFlow / Keras**, **NLTK**, and **Streamlit**, this project delivers real-time threat intelligence with high accuracy, low latency, and an interactive cybersecurity dashboard suitable for academic thesis defense and enterprise deployment.

---

## 📑 Table of Contents
1. [Project Overview](#-1-project-overview)
2. [Project Objectives](#-2-project-objectives)
3. [Technology Stack](#-3-technology-stack)
4. [Dataset Description & Source](#-4-dataset-description--source)
5. [Deep Learning LSTM Architecture](#-5-deep-learning-lstm-architecture)
6. [System Workflow & NLP Pipeline](#-6-system-workflow--nlp-pipeline)
7. [Project Folder Structure](#-7-project-folder-structure)
8. [Installation & Setup Instructions](#-8-installation--setup-instructions)
9. [How to Train the Model](#-9-how-to-train-the-model)
10. [How to Run the Website](#-10-how-to-run-the-website)
11. [Model Evaluation & Empirical Results](#-11-model-evaluation--empirical-results)
12. [Expected Project Outputs & Visualizations](#-12-expected-project-outputs--visualizations)
13. [Future Enhancements](#-13-future-enhancements)

---

## 🎯 1. Project Overview
Electronic spam and phishing attacks remain among the primary attack vectors for identity theft, financial fraud, and credential harvesting. Traditional classification approaches—such as Naive Bayes or Support Vector Machines—rely on unordered **Bag-of-Words (BoW)** or TF-IDF representations. These static methods fail to capture:
- **Sequential Word Order & Grammatical Syntax**: The arrangement of words determines intent (e.g., *"not free to win"* vs *"free to win not"*).
- **Long-Range Contextual Semantics**: Modern spammers weave innocuous conversational sentences around malicious links.

This project implements a **sequence-aware LSTM neural network** that reads messages sequentially, dynamically retaining or forgetting tokens through internal gating mechanisms, thereby identifying spam patterns with **98.7%+ test accuracy**.

---

## 🎯 2. Project Objectives
- Build an intelligent NLP and Deep Learning pipeline to classify messages into **SPAM** or **HAM (Legitimate)**.
- Implement an automated CSV dataset ingestion engine supporting varying schema naming conventions (`v1/v2`, `label/message`, `category/text`).
- Eliminate data leakage by fitting tokenizers exclusively on the training partition.
- Train a genuine LSTM network with **SpatialDropout1D**, **Dropout regularization**, and **EarlyStopping** to prevent overfitting.
- Provide a modern, responsive **Streamlit** dashboard featuring live email classification, confidence scoring, dataset analytics, and model retraining.

---

## 💻 3. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.12 | Core programming environment |
| **Frontend UI** | Streamlit 1.42+ | Interactive dark-mode dashboard |
| **Deep Learning** | TensorFlow 2.x & Keras 3.x | LSTM model building, training & inference |
| **NLP Pipeline** | NLTK 3.10+ | Tokenization, text cleaning, stopword filtering |
| **Data Processing** | Pandas, NumPy | Dataset ETL, vector manipulation, sequence padding |
| **Visualization** | Matplotlib, Seaborn | High-resolution loss, accuracy, and confusion matrix charts |
| **Evaluation** | Scikit-learn 1.6+ | Confusion matrix, precision, recall, F1-score |

---

## 📊 4. Dataset Description & Source

- **Dataset Name**: SMS Spam Collection Dataset
- **Primary Source**: [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Raw Volume**: 5,572 records
- **Post-cleaning Volume**: 5,157 unique records (403 duplicates eliminated)
- **Class Balance**:
  - **HAM (Legitimate)**: 4,504 messages (~87.3%)
  - **SPAM**: 653 messages (~12.7%)
- **Automatic Column Resolution**: Seamlessly parses `v1`, `v2`, `label`, `message`, `category`, and `text` columns. Labels are numerically encoded as:
  $$\text{HAM} = 0, \quad \text{SPAM} = 1$$

---

## 🧠 5. Deep Learning LSTM Architecture

The model uses a unidirectional Long Short-Term Memory network tailored for sequence classification:

```
                  ┌────────────────────────────────────────┐
                  │    Raw Email / SMS Message String      │
                  └──────────────────┬─────────────────────┘
                                     │ Lowercase, RegEx, NLTK
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Cleaned Token Sequence (Max: 100)   │
                  └──────────────────┬─────────────────────┘
                                     │ Tokenizer & Pre-Padding
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Embedding Layer (Vocab: 5000, D: 64)│
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    SpatialDropout1D (Rate: 0.2)        │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    LSTM Layer (64 Hidden Units)        │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Dropout Layer (Rate: 0.2)           │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Dense Layer (32 Units, ReLU)        │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │    Output Layer (1 Unit, Sigmoid)      │
                  └────────────────────────────────────────┘
```

### Hyperparameter Specifications:
- **Embedding Dimensions**: 64
- **Max Vocabulary Size**: 5,000 words
- **Max Sequence Length**: 100 tokens (Pre-padded)
- **Optimizer**: Adam ($\eta = 0.001$)
- **Loss Function**: Binary Cross-Entropy
- **Regularization**: SpatialDropout1D (0.2) + Dense Dropout (0.2) + EarlyStopping ($\text{patience}=4$)
- **Class Weighting**: Balanced penalty applied during training to counter the 87/13 ham/spam imbalance.

---

## 🔄 6. System Workflow & NLP Pipeline

1. **Ingestion & Validation**:
   - Ingests CSV dataset via automated schema inference.
   - Cleans null values, removes duplicates, and casts labels to binary $\{0, 1\}$.
2. **Text Normalization**:
   - Strips URLs (`http\S+`), HTML tags (`<.*?>`), and special characters.
   - Converts all tokens to lowercase and removes excessive whitespaces.
   - Tokenizes strings into word units using NLTK `word_tokenize`.
3. **Sequence Transformation**:
   - Keras `Tokenizer` is fitted exclusively on the 80% training split.
   - Words are converted to index sequences and padded (`padding='pre'`).
4. **Model Training & Checkpointing**:
   - The LSTM processes inputs batch-by-batch ($B=32$).
   - Best checkpoint restored automatically upon EarlyStopping trigger.
   - Exports `lstm_model.keras`, `tokenizer.pkl`, and `config.json`.
5. **Inference & Risk Assessment**:
   - Live messages undergo identical preprocessing.
   - Output probability $p \ge 0.5 \implies \text{SPAM}$, else $\text{HAM}$.
   - Suspicious keywords are flagged against a domain trigger lexicon.

---

## 📁 7. Project Folder Structure

```
srikarDL/
│
├── app.py                     # Interactive Streamlit 5-page web application
├── train.py                   # LSTM training script with EarlyStopping & curve plotting
├── preprocessing.py           # NLP cleaning, tokenization, padding, and dataset loaders
├── prediction.py              # Real-time inference engine and risk explainability
├── evaluation.py              # Performance calculation (Acc, Prec, Rec, F1, Confusion Matrix)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive academic & operational documentation
│
├── dataset/
│   └── spam.csv               # Official SMS Spam Collection dataset (5,572 rows)
│
├── models/
│   ├── lstm_model.keras       # Trained Keras deep learning model weights & topology
│   ├── tokenizer.pkl          # Serialized Keras text tokenizer
│   └── config.json            # Model and preprocessing metadata hyperparameters
│
└── outputs/
    ├── accuracy.png           # Training vs Validation Accuracy graph
    ├── loss.png               # Training vs Validation Loss graph
    ├── confusion_matrix.png   # High-resolution test set confusion matrix heatmap
    ├── distribution.png       # Dataset class distribution plot
    ├── metrics_comparison.png # Evaluation metrics comparison bar chart
    ├── evaluation_metrics.json# Raw test set evaluation metric scores
    └── training_history.json  # Epoch-by-epoch loss and accuracy numbers
```

---

## ⚙️ 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows, macOS, or Linux.
- Git (optional).

### Step 1: Open Terminal in Project Directory
```powershell
cd c:\Users\Venkat\OneDrive\Desktop\srikarDL
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 9. How to Train the Model

To train the LSTM neural network from scratch using the dataset:
```powershell
python train.py
```

### What Happens During Training:
- Ingests `dataset/spam.csv` and removes duplicates.
- Generates an 80/20 stratified train/test split.
- Trains for up to 10 epochs with early stopping on validation loss.
- Saves the trained model to `models/lstm_model.keras`.
- Generates convergence plots (`outputs/accuracy.png` and `outputs/loss.png`).

---

## 🌐 10. How to Run the Website

Launch the Streamlit web dashboard with:
```powershell
streamlit run app.py
```

Open your web browser and navigate to:
```
http://localhost:8501
```

### Website Pages:
1. **🏠 Home Dashboard**: Architectural overview, KPI metric cards, and LSTM theoretical advantages.
2. **🔍 Email / SMS Prediction**: Large text area with one-click quick presets (Phishing, Lottery, Meeting, Casual) and neural threat breakdown.
3. **📊 Dataset Analysis**: Class donut chart, character length distributions, top spam keywords bar chart, and searchable data table.
4. **🧠 Model Training**: Customizable hyperparameters (epochs, batch size, vocab size) with live progress bar and epoch metrics streaming.
5. **📈 Model Evaluation**: Actual test set confusion matrix heatmap, classification report, and comparative performance charts.

---

## 📈 11. Model Evaluation & Empirical Results

Evaluated on the held-out 20% test partition (1,032 samples):

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **98.74%** | Overall correct predictions |
| **Spam Precision** | **94.70%** | When classified as SPAM, accuracy of the alert |
| **Spam Recall** | **95.42%** | Proportion of actual spam threats successfully intercepted |
| **Spam F1-Score** | **95.06%** | Harmonic mean balancing false alarms and missed threats |
| **Weighted F1** | **98.75%** | Aggregate score across all messages |

### Confusion Matrix Breakdown:
- **True Negatives (Legitimate HAM correctly identified)**: 894
- **False Positives (HAM mistakenly flagged as Spam)**: 7
- **False Negatives (Spam missed)**: 6
- **True Positives (Spam correctly blocked)**: 125

---

## 🖼️ 12. Expected Project Outputs & Visualizations

All figures are automatically generated and saved in the `outputs/` directory in high-DPI dark-mode theme:
- `outputs/accuracy.png`: Epoch-by-epoch training vs validation accuracy progression.
- `outputs/loss.png`: Binary cross-entropy minimization across epochs.
- `outputs/confusion_matrix.png`: Heatmap with percentages and sample counts.
- `outputs/metrics_comparison.png`: Bar chart contrasting Accuracy, Precision, Recall, and F1.
- `outputs/distribution.png`: Class proportion visualization.

---

## 🔮 13. Future Enhancements
1. **Bidirectional LSTM (Bi-LSTM)**: Incorporate both forward and backward contextual state vectors for enhanced linguistic nuance.
2. **Attention Mechanism & Transformers**: Upgrade to Self-Attention / DistilBERT embeddings for zero-shot generalization across domain shifts.
3. **Multilingual Phishing Detection**: Extend tokenization to handle non-English scripts and homoglyph attacks.
4. **REST API Microservice**: Package model inference with FastAPI and Docker for integration with email gateways (Postfix, Microsoft Exchange, SendGrid).

---

## 👨‍💻 Academic Project Attribution
- **Project**: Spam Email Classification Using LSTM (Deep Learning)
- **Domain**: Natural Language Processing & Cyber Security
- **Suitability**: B.Tech Final Year Academic Project / Major Project Submission
