# 🚀 Hybrid Sentiment Intelligence System

A **dynamic, explainable sentiment analysis system** that intelligently combines:

- ⚡ Rule-based NLP (VADER)
- 🧠 Deep Learning (DistilBERT)

Using a **dynamic alpha weighting mechanism** to adapt based on input text.

---

## 🔥 Key Features

- 🧠 **Hybrid AI Model** (VADER + BERT)
- 🎯 **Dynamic Alpha Weighting**
- 🌐 **Hinglish & Noisy Text Support**
- 📊 **Interactive Dashboard (Plotly)**
- 📁 **CSV Batch Analysis**
- 📄 **PDF Sentiment Extraction**
- 🔍 **Explainable AI (Model Contribution Visualized)**

---

## 🧠 How It Works

The system dynamically decides which model to trust:

| Input Type        | Model Priority |
|------------------|---------------|
| Short / Noisy    | VADER         |
| Long / Clean     | DistilBERT    |

### Hybrid Formula: Final Score = α * VADER + (1 - α) * BERT


Where:
- α is computed using **text length + noise level**

---

## 📊 Dashboard Features

- 📈 Score comparison (VADER vs BERT vs Final)
- 🥧 Model contribution visualization
- 📉 Alpha distribution (CSV)
- 🚨 Detection of highly negative text

---

## 📁 File Intelligence

### CSV Support
- Auto-detect text column
- Batch sentiment analysis
- Insights & visualization

### PDF Support
- Extracts text from document
- Sentence-level sentiment analysis
- Highlights most negative lines

---

## 🛠️ Tech Stack

- Python
- Streamlit
- HuggingFace Transformers
- VADER Sentiment
- Plotly
- Pandas
- PyPDF2

---

🌐 Live Demo : https://dynamic-hybrid-sentiment-analysis-tech-with-jhalak.streamlit.app/
