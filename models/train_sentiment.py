"""
Trains the TF-IDF + Logistic Regression sentiment baseline on the Olist
review text and pickles it for the backend to serve.

An optional heavier upgrade path (BERTimbau, a Portuguese BERT model) is
sketched at the bottom of this file for teams that have torch/transformers
available and want higher accuracy on nuanced/sarcastic reviews.

Run from the project root:
    python models/train_sentiment.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_pipeline import load_and_clean
from utils.sentiment_analysis import train_baseline_sentiment_model

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "olist_master_dataset.csv"
MODEL_DIR = ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("SENTIMENT ANALYSIS — TRAINING PIPELINE")
    print("=" * 70)

    print("[1] Loading & cleaning raw data...")
    df = load_and_clean(str(DATA_PATH))

    # dedupe to one row per review
    reviews = df.dropna(subset=["review_id"]).drop_duplicates(subset=["review_id"]).copy()
    print(f"    Unique reviews: {reviews.shape[0]}")
    print(f"    Sentiment distribution:\n{reviews['sentiment'].value_counts()}")

    print("\n[2] Training TF-IDF + Logistic Regression baseline...")
    vectorizer, model, report = train_baseline_sentiment_model(reviews)

    print("\n[3] Classification report (holdout test set):")
    print(pd.DataFrame(report).T.round(4).to_string())

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")

    # ---- Business-facing aggregates for the dashboard/insights layer -----
    labeled = reviews[reviews["sentiment"] != "Unknown"]
    summary = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_reviews": int(len(labeled)),
        "sentiment_distribution": labeled["sentiment"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "avg_review_score": round(float(labeled["review_score"].mean()), 2),
        "classification_report": report,
        "sentiment_by_delivery_status": (
            pd.crosstab(labeled["delivery_status"], labeled["sentiment"], normalize="index")
            .mul(100).round(2).to_dict(orient="index")
        ),
        "sentiment_by_top_category": (
            labeled[labeled["product_category_name_english"].isin(
                ["bed_bath_table", "health_beauty", "sports_leisure", "watches_gifts",
                 "computers_accessories", "furniture_decor", "housewares", "telephony",
                 "auto", "garden_tools"])]
            .groupby("product_category_name_english")["sentiment"]
            .value_counts(normalize=True).mul(100).round(2)
            .unstack(fill_value=0).to_dict(orient="index")
        ),
        "sentiment_by_state_top10": (
            pd.crosstab(labeled["customer_state"], labeled["sentiment"], normalize="index")
            .mul(100).round(2)
            .loc[labeled["customer_state"].value_counts().head(10).index]
            .to_dict(orient="index")
        ),
    }
    with open(MODEL_DIR / "sentiment_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[4] Saved artifacts to {MODEL_DIR}/")
    print("    - tfidf_vectorizer.pkl")
    print("    - sentiment_model.pkl")
    print("    - sentiment_metadata.json")
    print("\nDone.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# OPTIONAL UPGRADE: BERTimbau transformer fine-tune (requires torch,
# transformers, spacy — not installed in this environment). Kept here as
# a reference for teams that want to swap in a stronger model later.
# ---------------------------------------------------------------------------
BERTIMBAU_SNIPPET = '''
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# tokenize df["clean_review"], wrap in a torch Dataset, then:
training_args = TrainingArguments(
    output_dir="./results", num_train_epochs=2,
    per_device_train_batch_size=32, per_device_eval_batch_size=32,
    eval_strategy="epoch", save_strategy="epoch",
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
trainer.train()
trainer.save_model("models/artifacts/bertimbau_sentiment")
'''
