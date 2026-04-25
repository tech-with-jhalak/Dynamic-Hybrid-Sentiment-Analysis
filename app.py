import os
import warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from hybrid_model import hybrid_predict

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Hybrid Sentiment AI", layout="wide")

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>

/* BACKGROUND */
body {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
    color: #e2e8f0;
}

/* TITLE */
.title {
    font-size: 44px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* CARD */
.card {
    padding: 25px;
    border-radius: 16px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 25px rgba(99,102,241,0.3);
    transition: 0.3s ease;
    margin-top: 20px;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 40px rgba(139,92,246,0.6);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 16px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(139,92,246,0.8);
}

/* INPUT */
textarea {
    background: rgba(255,255,255,0.05) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(99,102,241,0.3);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">Dynamic Hybrid Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable AI • VADER + DistilBERT</div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔍 Single Analysis", "📁 File Analysis"])

# ================= SINGLE =================
with tab1:

    text = st.text_area("Enter text", height=120)

    if st.button("Analyze 🚀"):

        if not text.strip():
            st.warning("Enter valid text")
        else:
            result = hybrid_predict(text)

            color = "#22c55e" if result["prediction"] == "POSITIVE" else "#ef4444"

            # RESULT CARD
            st.markdown(f"""
            <div class="card">
                <h2 style="text-align:center; color:{color};">
                {result['prediction']}
                </h2>
            </div>
            """, unsafe_allow_html=True)

            # GLOW LINE
            st.markdown("""
            <div style="
            height:4px;
            background: linear-gradient(90deg,#60a5fa,#8b5cf6);
            border-radius:5px;
            margin-top:10px;
            animation: glow 2s infinite alternate;">
            </div>

            <style>
            @keyframes glow {
                from { box-shadow: 0 0 5px #6366f1; }
                to { box-shadow: 0 0 20px #8b5cf6; }
            }
            </style>
            """, unsafe_allow_html=True)

            # METRICS
            col1, col2, col3 = st.columns(3)
            col1.metric("VADER Weight", result["alpha"])
            col2.metric("BERT Weight", round(1-result["alpha"],3))
            col3.metric("Confidence", result["confidence"])

            # ---------------- PIE ----------------
            st.markdown("### 📊 Model Contribution")

            pie = go.Figure(data=[go.Pie(
                labels=["VADER", "BERT"],
                values=[result["alpha"], 1-result["alpha"]],
                hole=0.5,
                marker=dict(colors=["#05e6fbe1", "#3476da"])
            )])

            pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(pie, use_container_width=True)

            # ---------------- BAR ----------------
            st.markdown("### 📈 Score Comparison")

            bar = go.Figure(data=[go.Bar(
                x=["VADER", "BERT", "FINAL"],
                y=[result["vader"], result["bert"], result["final_score"]],
                marker_color=["#60a5fa", "#a78bfa", "#34d399"]
            )])

            bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white")
            )

            st.plotly_chart(bar, use_container_width=True)

            # EXPLANATION
            st.markdown("### 🧠 Explanation")

            st.info(f"""
            • {round(result["alpha"]*100,1)}% VADER  
            • {round((1-result["alpha"])*100,1)}% BERT  

            Length: {result["length"]}  
            Noise: {result["noise"]}  

            Model dynamically adjusts based on input
            """)

# ================= FILE =================
with tab2:

    uploaded = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])

    if uploaded:

        if uploaded.name.endswith(".csv"):

            df = pd.read_csv(uploaded)

            text_col = None
            for col in df.columns:
                if "text" in col.lower():
                    text_col = col
                    break

            if text_col is None:
                st.error("No text column found")
            else:
                results = [hybrid_predict(str(t)) for t in df[text_col]]
                out_df = pd.DataFrame(results)

                st.dataframe(out_df)

                st.markdown("### 📊 Insights")

                fig = go.Figure()
                fig.add_bar(
                    x=out_df["prediction"].value_counts().index,
                    y=out_df["prediction"].value_counts().values,
                    marker_color=["#60a5fa", "#4103fd"]
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )

                st.plotly_chart(fig, use_container_width=True)

        elif uploaded.name.endswith(".pdf"):

            reader = PdfReader(uploaded)
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            sentences = text.split(".")
            results = [hybrid_predict(s) for s in sentences if len(s.strip()) > 5]

            df = pd.DataFrame(results)

            st.dataframe(df)

            st.markdown("### 🚨 Most Negative Sentences")

            worst = df.sort_values("final_score").head(5)

            for _, row in worst.iterrows():
                st.error(row["text"])