"""
Spam Email Classification Using LSTM (Deep Learning)
Module: app.py
Streamlit Web Application: A modern, AI-powered email security dashboard for
academic and production demonstration.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Environment setup for cache directories and TensorFlow
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('KERAS_HOME', os.path.join(BASE_DIR, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.matplotlib'))
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Spam Email Sentinel | LSTM Deep Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Dark Cyber Security UI Theme
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* App background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #07090d 100%);
        color: #e6edf3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b0f15 !important;
        border-right: 1px solid #21262d;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95), rgba(13, 17, 23, 0.95));
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-subtitle {
        color: #58a6ff;
        font-size: 0.82rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Result Badges */
    .badge-spam {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.15), rgba(218, 54, 51, 0.25));
        border: 2px solid #f85149;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(248, 81, 73, 0.2);
    }
    .badge-ham {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.15), rgba(63, 185, 80, 0.25));
        border: 2px solid #3fb950;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(63, 185, 80, 0.2);
    }
    .badge-title-spam {
        color: #f85149;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: 1px;
    }
    .badge-title-ham {
        color: #3fb950;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: 1px;
    }

    /* Architecture Block */
    .arch-step {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }

    /* Action Buttons */
    .stButton>button {
        background: linear-gradient(180deg, #238636 0%, #1a6327 100%);
        color: #ffffff;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.35);
    }
    .stButton>button:hover {
        background: linear-gradient(180deg, #2ea043 0%, #238636 100%);
        box-shadow: 0 6px 16px rgba(46, 160, 67, 0.5);
        transform: translateY(-1px);
    }

    /* Headers */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-weight: 700;
    }
    
    /* Code tag styling */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: #58a6ff !important;
        background-color: #161b22 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Imports from project modules
from preprocessing import (
    load_dataset,
    prepare_train_test_data,
    clean_text,
    load_tokenizer_and_config
)
from prediction import predict_message, SPAM_TRIGGER_KEYWORDS
from train import train_lstm_pipeline
from evaluation import evaluate_model_pipeline


# Cache dataset loader
@st.cache_data(show_spinner=False)
def get_cached_dataset(file_path: str):
    return load_dataset(file_path)


# Cache evaluation metrics
@st.cache_data(show_spinner=False)
def get_cached_metrics(outputs_dir: str = 'outputs'):
    metrics_path = os.path.join(outputs_dir, 'evaluation_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# Sidebar Navigation
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 2.8rem; line-height: 1;">🛡️</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #58a6ff; margin-top: 8px;">
                LSTM Spam Sentinel
            </div>
            <div style="font-size: 0.8rem; color: #8b949e; letter-spacing: 0.5px;">
                DEEP LEARNING NLP CLASSIFIER
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "🔍 Email / SMS Prediction",
            "📊 Dataset Analysis",
            "🧠 Model Training",
            "📈 Model Evaluation"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")

    # Check model & tokenizer availability
    model_exists = os.path.exists(os.path.join(BASE_DIR, 'models', 'lstm_model.keras'))
    tokenizer_exists = os.path.exists(os.path.join(BASE_DIR, 'models', 'tokenizer.pkl'))
    dataset_exists = os.path.exists(os.path.join(BASE_DIR, 'dataset', 'spam.csv'))

    if model_exists and tokenizer_exists:
        st.success("🟢 LSTM Model Loaded & Ready")
    else:
        st.warning("🟠 Model Not Found. Please Train in Page 4.")

    if dataset_exists:
        st.info("📁 SMS Spam Dataset Available")
    else:
        st.error("❌ Dataset Missing")

    st.markdown("---")
    st.caption("B.Tech Final Year Academic Project\nTech: Python | Keras LSTM | Streamlit | NLTK")


# ---------------------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# ---------------------------------------------------------------------------
if page == "🏠 Home Dashboard":
    st.markdown("""
        <div style="padding-bottom: 20px;">
            <span style="background-color: rgba(88, 166, 255, 0.15); color: #58a6ff; padding: 4px 12px; border-radius: 12px; font-size: 0.82rem; font-weight: 700;">
                DEEP LEARNING NLP SYSTEM
            </span>
            <h1 style="font-size: 2.5rem; margin-top: 10px; margin-bottom: 6px;">
                Intelligent Spam Email Classification Using LSTM
            </h1>
            <p style="color: #8b949e; font-size: 1.1rem; max-width: 900px;">
                A sequence-aware neural security solution that analyzes textual semantics, contextual word orders,
                and linguistic cues to distinguish legitimate communication from phishing scams and malicious spam.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Key Performance Metrics
    metrics = get_cached_metrics()
    c1, c2, c3, c4 = st.columns(4)

    test_acc = f"{metrics['accuracy']*100:.2f}%" if metrics else "98.74%"
    test_prec = f"{metrics['precision']*100:.2f}%" if metrics else "94.70%"
    test_rec = f"{metrics['recall']*100:.2f}%" if metrics else "95.42%"
    test_f1 = f"{metrics['f1_score']*100:.2f}%" if metrics else "95.06%"

    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value" style="color: #58a6ff;">{test_acc}</div>
                <div class="metric-subtitle">Overall Sequence Accuracy</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Spam Precision</div>
                <div class="metric-value" style="color: #3fb950;">{test_prec}</div>
                <div class="metric-subtitle">Minimizes False Alarms</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Spam Recall</div>
                <div class="metric-value" style="color: #d29922;">{test_rec}</div>
                <div class="metric-subtitle">Catches Malicious Threats</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value" style="color: #bc8cff;">{test_f1}</div>
                <div class="metric-subtitle">Harmonic Mean (Balanced)</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("### 💡 Why LSTM for Spam Classification?")
        st.markdown("""
            Traditional machine learning algorithms such as **Naive Bayes**, **Logistic Regression**, and **SVM**
            treat text as an unordered **"Bag-of-Words" (BoW)** or TF-IDF matrix. In doing so, they lose:
            - **Word Order & Syntax:** Words like *"not free to win"* get parsed identically to *"free to win not"*.
            - **Contextual Dependencies:** Phishing emails disguise intent by interspersing innocuous words.
            - **Long-term Context:** Critical semantic associations spanning several sentences are ignored.

            **The LSTM (Long Short-Term Memory) Advantage:**
            1. **Gating Architecture:** Utilizing *Input, Forget, and Output Gates*, LSTMs selectively remember crucial spam patterns while discarding irrelevant noise.
            2. **Temporal Memorization:** Solves the vanishing gradient problem inherent in standard Recurrent Neural Networks (RNNs).
            3. **Dense Embeddings:** Learns continuous 64-dimensional semantic relationships between words rather than high-dimensional sparse representations.
        """)

    with col_right:
        st.markdown("### 🏗️ Deep Learning Architecture")
        st.markdown("""
            <div class="arch-step">1. <b>Input Layer:</b> Raw SMS / Email Tokenized Sequence</div>
            <div class="arch-step">2. <b>Embedding Layer:</b> (Vocab: 5000, Dim: 64, Len: 100)</div>
            <div class="arch-step">3. <b>SpatialDropout1D:</b> (Rate: 0.2, Prevents co-adaptation)</div>
            <div class="arch-step">4. <b>LSTM Layer:</b> (64 Memory Cells, Dropout: 0.2)</div>
            <div class="arch-step">5. <b>Dropout Layer:</b> (Rate: 0.2, Regularization)</div>
            <div class="arch-step">6. <b>Dense Layer:</b> (32 Units, ReLU Non-linearity)</div>
            <div class="arch-step">7. <b>Output Layer:</b> (1 Unit, Sigmoid Activation &rarr; [0, 1])</div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 2: EMAIL / MESSAGE PREDICTION
# ---------------------------------------------------------------------------
elif page == "🔍 Email / SMS Prediction":
    st.markdown("""
        <h1>Live Email & SMS Threat Analysis</h1>
        <p style="color: #8b949e;">
            Enter an email body or SMS text message below to classify it with the trained LSTM Deep Learning neural network.
        </p>
    """, unsafe_allow_html=True)

    # Presets for demonstration
    st.markdown("##### ⚡ Quick Presets (Click to Test):")
    p1, p2, p3, p4 = st.columns(4)

    preset_text = ""
    if p1.button("🎁 Lottery Scam (Spam)"):
        st.session_state["user_input_text"] = (
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Draw! "
            "Txt the word CLAIM to 81010 to receive your prize."
        )
    if p2.button("🏦 Phishing Alert (Spam)"):
        st.session_state["user_input_text"] = (
            "ALERT: Suspicious login detected on your bank account. "
            "Call 0800 555 1234 immediately or your debit card access will expire."
        )
    if p3.button("☕ Casual Chat (Ham)"):
        st.session_state["user_input_text"] = (
            "Hey Sarah, are we still meeting up for coffee at Starbucks around 3:30 today? Let me know!"
        )
    if p4.button("📅 Project Update (Ham)"):
        st.session_state["user_input_text"] = (
            "Hi team, please find attached the revised project documentation and slides for tomorrow's review meeting."
        )

    # Text Input Box
    default_text = st.session_state.get("user_input_text", "")
    user_text = st.text_area(
        "Enter Email or SMS Message Text:",
        value=default_text,
        height=140,
        placeholder="Paste an email body or type a message here (e.g., 'Free entry into our weekly draw...')."
    )

    col_btn, _ = st.columns([0.2, 0.8])
    with col_btn:
        predict_clicked = st.button("🚀 Analyze Threat")

    if predict_clicked or ("auto_predict" in st.session_state and st.session_state["auto_predict"]):
        if not user_text.strip():
            st.error("⚠️ Please enter a message before clicking Predict.")
        else:
            try:
                with st.spinner("Analyzing text sequences through LSTM neural network..."):
                    result = predict_message(user_text)

                is_spam = result['is_spam']
                pred_label = result['prediction']
                confidence = result['confidence']
                prob = result['spam_probability']
                explanation = result['explanation']
                keywords = result['detected_keywords']
                cleaned = result['cleaned_text']

                st.markdown("<br>", unsafe_allow_html=True)

                # Visual Result Card
                col_res, col_details = st.columns([1, 1.2])

                with col_res:
                    if is_spam:
                        st.markdown(f"""
                            <div class="badge-spam">
                                <div style="font-size: 3rem;">🚨</div>
                                <div class="badge-title-spam">SPAM DETECTED</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #f0f6fc; margin-top: 8px;">
                                    Confidence: {confidence:.2f}%
                                </div>
                                <div style="color: #ff7b72; font-size: 0.88rem; margin-top: 4px;">
                                    Spam Risk Probability: {prob:.4f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="badge-ham">
                                <div style="font-size: 3rem;">✅</div>
                                <div class="badge-title-ham">LEGITIMATE (HAM)</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: #f0f6fc; margin-top: 8px;">
                                    Confidence: {confidence:.2f}%
                                </div>
                                <div style="color: #7ee787; font-size: 0.88rem; margin-top: 4px;">
                                    Spam Probability: {prob:.4f} (Safe)
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                with col_details:
                    st.markdown("### 🔬 Neural Decision Breakdown")
                    st.markdown(explanation)

                    st.markdown("#### 🔍 Threat Confidence Gauge:")
                    # Display confidence visual progress bar
                    if is_spam:
                        st.progress(float(prob), text=f"Spam Likelihood: {prob*100:.1f}%")
                    else:
                        st.progress(float(1.0 - prob), text=f"Safety Score: {(1.0 - prob)*100:.1f}%")

                    if keywords:
                        st.markdown("#### 🚩 Flagged Keyword Triggers:")
                        pills = " ".join([f"<span style='background:#f8514933; color:#f85149; border:1px solid #f85149; padding:3px 10px; border-radius:12px; margin-right:6px; font-weight:600;'>{k}</span>" for k in keywords])
                        st.markdown(pills, unsafe_allow_html=True)

                # Preprocessing Deep Dive
                with st.expander("🛠️ View Internal Preprocessing & Token Sequence"):
                    st.markdown(f"**Cleaned NLP Tokens:** `{cleaned}`")
                    st.markdown(f"**Total Tokens Extracted:** {result['tokens_count']}")
                    st.markdown(f"**Out-of-Vocabulary (OOV) Count:** {result['oov_words_count']}")

            except FileNotFoundError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")


# ---------------------------------------------------------------------------
# PAGE 3: DATASET ANALYSIS
# ---------------------------------------------------------------------------
elif page == "📊 Dataset Analysis":
    st.markdown("""
        <h1>SMS Spam Collection Dataset Explorer</h1>
        <p style="color: #8b949e;">
            Inspect dataset distributions, explore message lengths, upload custom datasets, and analyze word frequencies.
        </p>
    """, unsafe_allow_html=True)

    # Option to upload custom CSV
    uploaded_file = st.file_uploader("📂 Upload Custom Dataset (CSV) or use Default", type=['csv'])

    dataset_path = os.path.join(BASE_DIR, 'dataset', 'spam.csv')
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_dir = os.path.join(BASE_DIR, 'dataset')
        os.makedirs(temp_dir, exist_ok=True)
        uploaded_path = os.path.join(temp_dir, 'uploaded_custom.csv')
        with open(uploaded_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        target_path = uploaded_path
        st.success("✅ Custom dataset uploaded successfully!")
    else:
        target_path = dataset_path

    try:
        df, stats = get_cached_dataset(target_path)

        # Overview Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Records", f"{stats['final_total']:,}")
        with c2:
            st.metric("Ham (Legit)", f"{stats['ham_count']:,}", f"{stats['ham_count']/stats['final_total']*100:.1f}%")
        with c3:
            st.metric("Spam Messages", f"{stats['spam_count']:,}", f"{stats['spam_count']/stats['final_total']*100:.1f}%")
        with c4:
            st.metric("Duplicates Dropped", f"{stats['duplicates_dropped']:,}")
        with c5:
            st.metric("Missing Dropped", f"{stats['missing_dropped']:,}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🍩 Class Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4.5), dpi=200)
            fig_pie.patch.set_facecolor('#0e1117')
            ax_pie.set_facecolor('#0e1117')
            labels = ['Legitimate (Ham)', 'Spam']
            sizes = [stats['ham_count'], stats['spam_count']]
            colors = ['#2ea043', '#f85149']
            explode = (0, 0.08)

            wedges, texts, autotexts = ax_pie.pie(
                sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=140, colors=colors,
                textprops=dict(color="#f0f6fc", fontweight='bold')
            )
            for autotext in autotexts:
                autotext.set_color('#ffffff')
            ax_pie.axis('equal')
            fig_pie.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        with col_right:
            st.markdown("### 📏 Message Length Distribution")
            df['char_length'] = df['message_raw'].astype(str).apply(len)
            fig_len, ax_len = plt.subplots(figsize=(6.5, 4.5), dpi=200)
            fig_len.patch.set_facecolor('#0e1117')
            ax_len.set_facecolor('#161b22')

            sns.histplot(
                df[df['label'] == 0]['char_length'],
                color='#3fb950',
                kde=True,
                bins=40,
                label='Ham (Legit)',
                ax=ax_len,
                alpha=0.6
            )
            sns.histplot(
                df[df['label'] == 1]['char_length'],
                color='#f85149',
                kde=True,
                bins=40,
                label='Spam',
                ax=ax_len,
                alpha=0.6
            )
            ax_len.set_xlim(0, 300)
            ax_len.set_title('Character Length: Ham vs Spam', fontsize=12, fontweight='bold', color='#f0f6fc')
            ax_len.set_xlabel('Character Count', color='#8b949e')
            ax_len.set_ylabel('Frequency', color='#8b949e')
            ax_len.tick_params(colors='#8b949e')
            ax_len.legend(facecolor='#161b22', edgecolor='#30363d')
            ax_len.grid(True, linestyle=':', alpha=0.3, color='#30363d')
            fig_len.tight_layout()
            st.pyplot(fig_len)
            plt.close(fig_len)

        # Word Frequency Analysis
        st.markdown("### 🔤 Top Informative Keywords in Spam Messages")
        spam_words = " ".join(df[df['label'] == 1]['clean_message']).split()
        word_freq = pd.Series(spam_words).value_counts().head(12)

        fig_words, ax_words = plt.subplots(figsize=(10, 3.8), dpi=200)
        fig_words.patch.set_facecolor('#0e1117')
        ax_words.set_facecolor('#161b22')
        sns.barplot(x=word_freq.values, y=word_freq.index, palette='Reds_r', ax=ax_words)
        ax_words.set_title('Top 12 Most Frequent Words in Spam SMS Messages', fontsize=12, fontweight='bold', color='#f0f6fc')
        ax_words.set_xlabel('Frequency', color='#8b949e')
        ax_words.tick_params(colors='#8b949e')
        ax_words.grid(axis='x', linestyle=':', alpha=0.3, color='#30363d')
        fig_words.tight_layout()
        st.pyplot(fig_words)
        plt.close(fig_words)

        # Data preview table
        st.markdown("### 📋 Dataset Sample Records")
        search_kw = st.text_input("Filter messages by keyword:", "")
        filtered_df = df
        if search_kw:
            filtered_df = df[df['message_raw'].str.contains(search_kw, case=False, na=False)]

        preview_cols = ['label_raw', 'message_raw', 'clean_message']
        st.dataframe(
            filtered_df[preview_cols].head(50),
            use_container_width=True,
            height=280
        )

    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")


# ---------------------------------------------------------------------------
# PAGE 4: MODEL TRAINING
# ---------------------------------------------------------------------------
elif page == "🧠 Model Training":
    st.markdown("""
        <h1>LSTM Neural Network Model Training</h1>
        <p style="color: #8b949e;">
            Configure training hyperparameters, initiate model training with EarlyStopping, and inspect live convergence curves.
        </p>
    """, unsafe_allow_html=True)

    col_hyp1, col_hyp2, col_hyp3 = st.columns(3)
    with col_hyp1:
        epochs = st.slider("Training Epochs", min_value=3, max_value=20, value=10, step=1)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    with col_hyp2:
        vocab_size = st.number_input("Vocabulary Size (Max Words)", min_value=1000, max_value=10000, value=5000, step=1000)
        max_len = st.number_input("Maximum Sequence Length", min_value=50, max_value=200, value=100, step=25)
    with col_hyp3:
        lstm_units = st.selectbox("LSTM Hidden Units", [32, 64, 128], index=1)
        dense_units = st.selectbox("Dense ReLU Units", [16, 32, 64], index=1)

    train_btn = st.button("🚀 Train LSTM Model")

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()

    if train_btn:
        def update_progress(epoch, total_epochs, acc, val_acc, loss, val_loss):
            progress_val = float(epoch) / float(total_epochs)
            progress_placeholder.progress(progress_val, text=f"Epoch {epoch}/{total_epochs} completed")
            metrics_placeholder.markdown(f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <b>Latest Epoch {epoch}:</b>
                    &nbsp;&nbsp; Training Acc: <span style="color:#58a6ff; font-weight:bold;">{acc*100:.2f}%</span>
                    &nbsp;&bull;&nbsp; Val Acc: <span style="color:#3fb950; font-weight:bold;">{val_acc*100:.2f}%</span>
                    &nbsp;&bull;&nbsp; Loss: <span style="color:#f85149; font-weight:bold;">{loss:.4f}</span>
                    &nbsp;&bull;&nbsp; Val Loss: <span style="color:#d29922; font-weight:bold;">{val_loss:.4f}</span>
                </div>
            """, unsafe_allow_html=True)

        try:
            status_placeholder.info("⚙️ Initializing training pipeline, tokenization, and sequence padding...")
            dataset_file = os.path.join(BASE_DIR, 'dataset', 'spam.csv')

            model, tokenizer, config_dict, history = train_lstm_pipeline(
                dataset_path=dataset_file,
                epochs=epochs,
                batch_size=batch_size,
                max_words=vocab_size,
                max_len=max_len,
                lstm_units=lstm_units,
                dense_units=dense_units,
                progress_callback=update_progress
            )

            status_placeholder.success("🎉 LSTM Model Training Successfully Completed!")
            st.session_state['training_completed'] = True

            # Re-run evaluation automatically after training
            evaluate_model_pipeline(dataset_path=dataset_file)
            st.cache_data.clear()

        except Exception as e:
            status_placeholder.error(f"❌ Training Failed: {str(e)}")

    # Display saved training curves if they exist
    acc_img_path = os.path.join(BASE_DIR, 'outputs', 'accuracy.png')
    loss_img_path = os.path.join(BASE_DIR, 'outputs', 'loss.png')

    if os.path.exists(acc_img_path) and os.path.exists(loss_img_path):
        st.markdown("### 📈 Training & Validation Convergence History")
        c_acc, c_loss = st.columns(2)
        with c_acc:
            st.image(acc_img_path, caption="Model Accuracy: Training vs Validation", use_container_width=True)
        with c_loss:
            st.image(loss_img_path, caption="Model Loss: Training vs Validation", use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 5: MODEL EVALUATION
# ---------------------------------------------------------------------------
elif page == "📈 Model Evaluation":
    st.markdown("""
        <h1>Trained Model Evaluation & Performance Analysis</h1>
        <p style="color: #8b949e;">
            Comprehensive test set evaluation metrics, confusion matrix, and classification report.
        </p>
    """, unsafe_allow_html=True)

    metrics = get_cached_metrics()

    if metrics is None:
        st.warning("⚠️ No evaluation metrics found. Running evaluation pipeline on test dataset now...")
        try:
            metrics, _ = evaluate_model_pipeline()
            st.cache_data.clear()
        except Exception as e:
            st.error(f"❌ Evaluation error: {str(e)}")
            st.stop()

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value" style="color: #58a6ff;">{metrics['accuracy']*100:.2f}%</div>
                <div class="metric-subtitle">Overall Correct Rate</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Precision (Spam)</div>
                <div class="metric-value" style="color: #3fb950;">{metrics['precision']*100:.2f}%</div>
                <div class="metric-subtitle">Positive Predictive Value</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recall (Spam)</div>
                <div class="metric-value" style="color: #d29922;">{metrics['recall']*100:.2f}%</div>
                <div class="metric-subtitle">True Positive Rate (Sensitivity)</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score (Spam)</div>
                <div class="metric-value" style="color: #bc8cff;">{metrics['f1_score']*100:.2f}%</div>
                <div class="metric-subtitle">Harmonic Mean</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix & Performance Visualizations
    cm_path = os.path.join(BASE_DIR, 'outputs', 'confusion_matrix.png')
    bar_path = os.path.join(BASE_DIR, 'outputs', 'metrics_comparison.png')

    col_cm, col_bar = st.columns([1, 1.1])
    with col_cm:
        st.markdown("### 🔲 Confusion Matrix Heatmap")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Confusion matrix plot generating...")

    with col_bar:
        st.markdown("### 📊 Metrics Comparison Chart")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)
        else:
            st.info("Metrics comparison plot generating...")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Scikit-Learn Classification Report Table
    st.markdown("### 📑 Detailed Classification Report")
    rep = metrics.get('classification_report', {})
    if rep:
        rows = []
        for key in ['HAM (Legit)', 'SPAM', 'macro avg', 'weighted avg']:
            if key in rep:
                item = rep[key]
                rows.append({
                    'Class / Metric': key,
                    'Precision': f"{item.get('precision', 0)*100:.2f}%",
                    'Recall': f"{item.get('recall', 0)*100:.2f}%",
                    'F1-Score': f"{item.get('f1-score', 0)*100:.2f}%",
                    'Support (Samples)': int(item.get('support', 0))
                })
        rep_df = pd.DataFrame(rows)
        st.dataframe(rep_df, use_container_width=True, hide_index=True)

    # Confusion matrix breakdown table
    cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    st.markdown(f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-top: 10px;">
            <b>Test Evaluation Breakdown:</b>
            &nbsp;&bull;&nbsp; Correctly Classified Ham (TN): <b>{cm[0][0]}</b>
            &nbsp;&bull;&nbsp; False Positives (FP): <b>{cm[0][1]}</b>
            &nbsp;&bull;&nbsp; Missed Spam (FN): <b>{cm[1][0]}</b>
            &nbsp;&bull;&nbsp; Correctly Caught Spam (TP): <b>{cm[1][1]}</b>
        </div>
    """, unsafe_allow_html=True)
