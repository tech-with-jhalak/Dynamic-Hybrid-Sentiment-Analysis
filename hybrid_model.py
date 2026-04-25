import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

vader = SentimentIntensityAnalyzer()

bert_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------------
# Noise Score
# -------------------------
def compute_noise(text):
    noise = 0

    # symbols / emojis
    noise += sum(1 for c in text if not c.isalnum() and not c.isspace())

    # repeated letters
    noise += sum(1 for word in text.split() if any(word.count(c) > 2 for c in set(word)))

    # Hinglish indicator
    hinglish_words = ["hai", "kya", "nahi", "acha", "yaar"]
    noise += sum(1 for word in text.lower().split() if word in hinglish_words)

    return noise


# -------------------------
# Length
# -------------------------
def compute_length(text):
    return len(text.split())


# -------------------------
# Dynamic Alpha
# -------------------------
def compute_alpha(L, N, w1=0.15, w2=0.6):
    z = w1 * (20 - L) + w2 * N
    return 1 / (1 + np.exp(-z))


# -------------------------
# Hinglish-aware VADER
# -------------------------
def vader_score(text):
    score = vader.polarity_scores(text)["compound"]

    hinglish_positive = ["acha", "mast", "badiya", "sahi"]
    hinglish_negative = ["bakwas", "ghatiya", "bekaar", "faltu"]

    text_lower = text.lower()

    for word in hinglish_positive:
        if word in text_lower:
            score += 0.4

    for word in hinglish_negative:
        if word in text_lower:
            score -= 0.7

    return max(min(score, 1), -1)


# -------------------------
# BERT Score
# -------------------------
def bert_score(text):
    result = bert_model(text)[0]
    return result["score"] if result["label"] == "POSITIVE" else -result["score"]


# -------------------------
# FINAL HYBRID
# -------------------------
def hybrid_predict(text):
    L = compute_length(text)
    N = compute_noise(text)

    alpha = compute_alpha(L, N)

    s_vader = vader_score(text)
    s_bert = bert_score(text)

    final_score = alpha * s_vader + (1 - alpha) * s_bert

    return {
        "text": text,
        "length": L,
        "noise": N,
        "alpha": round(alpha, 3),
        "vader": round(s_vader, 3),
        "bert": round(s_bert, 3),
        "final_score": round(final_score, 3),
        "prediction": "POSITIVE" if final_score > 0 else "NEGATIVE",
        "confidence": round(abs(final_score), 3)
    }