"""
Sentiment analysis engine.

Uses the TF-IDF + Logistic Regression baseline from
Capstone_Project_E_commerce_sentiment_Analysis1.ipynb (the BERTimbau
transformer variant is available as an optional heavier upgrade — see
models/train_sentiment.py — but requires torch/transformers/spacy to be
installed, which is not assumed here).
"""

import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Minimal built-in Portuguese stopword list (avoids an nltk download dependency).
PT_STOPWORDS = set("""
a ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela
delas dele deles depois do dos e ela elas ele eles em entre era eram essa
essas esse esses esta estas este estes eu foi fomos for foram fui há isso
isto já lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no
nos nossa nossas nosso nossos num numa não nós o os ou para pela pelas pelo
pelos por qual quando que quem se seu seus somos sua suas são só também te
tem temos teu teus tu tua tuas um uma você vocês vos à às
""".split())


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = [t for t in text.split() if t not in PT_STOPWORDS and len(t) > 1]
    return " ".join(tokens)


class SentimentFeatureEncoder:
    """Encodes the Pareto top-10 product categories for the sentiment model."""

    TOP_10_CATEGORIES = [
        "bed_bath_table", "health_beauty", "sports_leisure", "watches_gifts",
        "computers_accessories", "furniture_decor", "housewares", "telephony",
        "auto", "garden_tools",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for cat in self.TOP_10_CATEGORIES:
            df[f"cat_{cat}"] = (df["product_category_name_english"] == cat).astype(int)
        df["cat_other"] = (~df["product_category_name_english"].isin(self.TOP_10_CATEGORIES)).astype(int)
        return df


def train_baseline_sentiment_model(df: pd.DataFrame, text_col="review_text", label_col="sentiment"):
    """Trains TF-IDF + Logistic Regression on labeled (non-Unknown) reviews."""
    from sklearn.model_selection import train_test_split

    data = df[df[label_col] != "Unknown"].copy()
    data["clean_review"] = data[text_col].apply(clean_text)
    data = data[data["clean_review"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_review"], data[label_col], test_size=0.2, random_state=42, stratify=data[label_col]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    report = classification_report(y_test, preds, digits=4, output_dict=True)

    return vectorizer, model, report


def predict_sentiment(texts, vectorizer, model):
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    return model.predict(X), model.predict_proba(X)
