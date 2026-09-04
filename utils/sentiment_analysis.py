"""
Sentiment analysis engine.

Two models are supported:

1. **TF-IDF + Logistic Regression (baseline)** – lightweight, fast, no GPU
   needed. Trained by models/train_sentiment.py and pickled to
   models/artifacts/tfidf_vectorizer.pkl + sentiment_model.pkl.

2. **BERTimbau Transformer** – neuralmind/bert-base-portuguese-cased fine-tuned
   for 3-class (Positive / Neutral / Negative) Portuguese sentiment. Requires
   torch and transformers (both already installed in this environment).
   Artifacts saved to models/artifacts/bertimbau_sentiment/.
"""

import re
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Portuguese stopword list (no nltk download required)
# ---------------------------------------------------------------------------
PT_STOPWORDS = set("""
a ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela
delas dele deles depois do dos e ela elas ele eles em entre era eram essa
essas esse esses esta estas este estes eu foi fomos for foram fui há isso
isto já lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no
nos nossa nossas nosso nossos num numa não nós o os ou para pela pelas pelo
pelos por qual quando que quem se seu seus somos sua suas são só também te
tem temos teu teus tu tua tuas um uma você vocês vos à às
""".split())

# Sentiment label order (must match notebook label_encoder classes: Negative,
# Neutral, Positive sorted alphabetically)
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


# ---------------------------------------------------------------------------
# Text cleaning (shared by both models)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Feature encoder (category one-hot used by baseline model)
# ---------------------------------------------------------------------------
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


# ===========================================================================
# MODEL 1 — TF-IDF + Logistic Regression baseline
# ===========================================================================
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
    """Run TF-IDF baseline inference and return (predictions, probabilities)."""
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    return model.predict(X), model.predict_proba(X)


# ===========================================================================
# MODEL 2 — BERTimbau transformer
# ===========================================================================
BERTIMBAU_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


def load_bertimbau_model_and_tokenizer(model_dir=None, num_labels=3):
    """
    Load BERTimbau tokenizer and model.

    Parameters
    ----------
    model_dir : str or Path, optional
        Path to a locally fine-tuned model directory. When None, loads the
        pretrained base weights from Hugging Face Hub.
    num_labels : int
        Number of output classes (default 3: Negative, Neutral, Positive).

    Returns
    -------
    model, tokenizer
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers must be installed to use BERTimbau. "
            "Run: pip install torch transformers"
        ) from exc

    source = str(model_dir) if model_dir else BERTIMBAU_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=num_labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_sentiment_bertimbau(texts, model, tokenizer, batch_size=16):
    """
    Run BERTimbau inference on a list of raw review texts.

    Parameters
    ----------
    texts : str or list[str]
        Raw (uncleaned) review text(s). Cleaning is applied internally.
    model : BertForSequenceClassification
        Loaded BERTimbau model.
    tokenizer : AutoTokenizer
    batch_size : int
        Number of texts to process per forward pass.

    Returns
    -------
    predictions : list[str]
        Predicted sentiment labels (``"Positive"``, ``"Neutral"``, ``"Negative"``).
    probabilities : np.ndarray, shape (n, 3)
        Softmax probabilities in the order [Negative, Neutral, Positive].
    """
    import torch
    import torch.nn.functional as F

    if isinstance(texts, str):
        texts = [texts]

    device = next(model.parameters()).device
    cleaned = [clean_text(t) for t in texts]

    all_probs = []
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    # Label order from model: matches sorted alphabetical [Negative, Neutral, Positive]
    pred_indices = all_probs.argmax(axis=1)
    predictions = [SENTIMENT_LABELS[i] for i in pred_indices]
    return predictions, all_probs


# ---------------------------------------------------------------------------
# BERTimbau fine-tuning helper
# ---------------------------------------------------------------------------
class _SentimentDataset:
    """Minimal torch.utils.data.Dataset wrapper for review texts + labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        import torch
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def train_bertimbau_sentiment_model(
    df: pd.DataFrame,
    output_dir: str,
    text_col="review_text",
    label_col="sentiment",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
):
    """
    Fine-tune BERTimbau (neuralmind/bert-base-portuguese-cased) for
    3-class Portuguese sentiment classification.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled review data. Must contain *text_col* and *label_col*.
    output_dir : str
        Directory to save the fine-tuned model and tokenizer.
    text_col : str
        Column containing raw review text.
    label_col : str
        Column containing sentiment labels (Positive / Neutral / Negative).
        Rows labelled "Unknown" are dropped.
    num_train_epochs : int
    per_device_train_batch_size : int
    per_device_eval_batch_size : int

    Returns
    -------
    report : dict
        Classification report dict (sklearn format) on the held-out test set.
    """
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    try:
        from sklearn.metrics import f1_score
    except ImportError:
        pass

    data = df[df[label_col] != "Unknown"].copy()
    data["clean_review"] = data[text_col].apply(clean_text)
    data = data[data["clean_review"].str.len() > 0].reset_index(drop=True)

    label_encoder = LabelEncoder()
    data["label_id"] = label_encoder.fit_transform(data[label_col])
    # Ensure class order: Negative=0, Neutral=1, Positive=2 (alphabetical)
    label_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_review"],
        data["label_id"],
        test_size=0.2,
        random_state=42,
        stratify=data["label_id"],
    )

    print(f"    Training rows : {len(X_train):,}")
    print(f"    Test rows     : {len(X_test):,}")

    tokenizer = AutoTokenizer.from_pretrained(BERTIMBAU_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERTIMBAU_MODEL_NAME, num_labels=len(label_names)
    )

    print("    Tokenizing training data…")
    train_enc = tokenizer(
        list(X_train.astype(str)),
        truncation=True, padding=True, max_length=128, return_tensors="pt",
    )
    print("    Tokenizing test data…")
    test_enc = tokenizer(
        list(X_test.astype(str)),
        truncation=True, padding=True, max_length=128, return_tensors="pt",
    )

    train_dataset = _SentimentDataset(train_enc, y_train.reset_index(drop=True))
    test_dataset = _SentimentDataset(test_enc, y_test.reset_index(drop=True))

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"f1": f1_score(p.label_ids, preds, average="weighted")}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("    Starting BERTimbau fine-tuning…")
    trainer.train()

    # Evaluate on held-out test set
    predictions_output = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions_output.predictions, axis=1)
    true_labels = np.array(y_test.reset_index(drop=True))

    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )

    # Save model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"    BERTimbau model saved to: {output_dir}")

    return report
