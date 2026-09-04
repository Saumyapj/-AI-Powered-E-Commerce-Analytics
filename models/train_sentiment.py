"""
Trains sentiment models on the Olist review text and pickles them for the
backend to serve.

Two models are trained:

1. **TF-IDF + Logistic Regression baseline** — fast, lightweight, no GPU
   required. Artifacts: tfidf_vectorizer.pkl + sentiment_model.pkl

2. **BERTimbau Transformer** — neuralmind/bert-base-portuguese-cased fine-tuned
   for 3-class Portuguese sentiment (Positive / Neutral / Negative).
   Requires torch + transformers (already installed). This runs on CPU when
   no GPU is available, so it will take longer (15-60 min for the full dataset).
   Artifacts: models/artifacts/bertimbau_sentiment/ directory.

Run from the project root:
    python models/train_sentiment.py

Skip the transformer training (faster dev iteration):
    python models/train_sentiment.py --skip-bertimbau
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_pipeline import load_and_clean
from utils.sentiment_analysis import (
    train_baseline_sentiment_model,
    train_bertimbau_sentiment_model,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "olist_master_dataset.csv"
MODEL_DIR = ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BERTIMBAU_DIR = MODEL_DIR / "bertimbau_sentiment"


def main(skip_bertimbau: bool = False):
    print("=" * 70)
    print("SENTIMENT ANALYSIS — TRAINING PIPELINE")
    print("=" * 70)

    print("[1] Loading & cleaning raw data...")
    df = load_and_clean(str(DATA_PATH))

    # Dedupe to one row per review
    reviews = df.dropna(subset=["review_id"]).drop_duplicates(subset=["review_id"]).copy()
    print(f"    Unique reviews: {reviews.shape[0]:,}")
    print(f"    Sentiment distribution:\n{reviews['sentiment'].value_counts()}")

    # ------------------------------------------------------------------
    # MODEL 1: TF-IDF + Logistic Regression baseline
    # ------------------------------------------------------------------
    print("\n[2] Training TF-IDF + Logistic Regression baseline...")
    vectorizer, model, baseline_report = train_baseline_sentiment_model(reviews)

    print("\n    Classification report (holdout test set):")
    print(pd.DataFrame(baseline_report).T.round(4).to_string())

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")
    print(f"    Saved: tfidf_vectorizer.pkl, sentiment_model.pkl")

    # ------------------------------------------------------------------
    # MODEL 2: BERTimbau transformer 
    # ------------------------------------------------------------------
    bertimbau_report = None
    if skip_bertimbau:
        print("\n[3] Skipping BERTimbau fine-tuning (--skip-bertimbau flag set).")
        print("    To train BERTimbau: run without --skip-bertimbau")
        # Load previous report if it already exists
        existing_meta = MODEL_DIR / "sentiment_metadata.json"
        if existing_meta.exists():
            with open(existing_meta) as f:
                prev = json.load(f)
            bertimbau_report = prev.get("bertimbau_classification_report")
    else:
        print("\n[3] Fine-tuning BERTimbau transformer...")
        print(f"    Model: neuralmind/bert-base-portuguese-cased")
        print(f"    Output dir: {BERTIMBAU_DIR}")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"    Device: {device.upper()}")
            BERTIMBAU_DIR.mkdir(parents=True, exist_ok=True)
            bertimbau_report = train_bertimbau_sentiment_model(
                reviews,
                output_dir=str(BERTIMBAU_DIR),
                num_train_epochs=2,
                per_device_train_batch_size=32,
                per_device_eval_batch_size=32,
            )
            print("\n    BERTimbau Classification Report (holdout test set):")
            print(pd.DataFrame(bertimbau_report).T.round(4).to_string())
        except Exception as exc:
            print(f"\n    [WARNING] BERTimbau training failed: {exc}")
            print("    Falling back to baseline-only mode.")
            bertimbau_report = None

    # ------------------------------------------------------------------
    # Business-facing aggregates + metadata JSON
    # ------------------------------------------------------------------
    labeled = reviews[reviews["sentiment"] != "Unknown"]

    # Build a side-by-side benchmark summary
    def _fmt_report(report):
        """Extract top-line metrics from a classification_report dict."""
        if report is None:
            return None
        wa = report.get("weighted avg", {})
        return {
            "accuracy": round(report.get("accuracy", 0), 4),
            "weighted_precision": round(wa.get("precision", 0), 4),
            "weighted_recall": round(wa.get("recall", 0), 4),
            "weighted_f1": round(wa.get("f1-score", 0), 4),
            "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
            "support": int(wa.get("support", 0)),
        }

    summary = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_reviews": int(len(labeled)),
        "sentiment_distribution": (
            labeled["sentiment"].value_counts(normalize=True)
            .mul(100).round(2).to_dict()
        ),
        "avg_review_score": round(float(labeled["review_score"].mean()), 2),

        # Baseline (TF-IDF + LR)
        "classification_report": baseline_report,
        "baseline_classification_report": baseline_report,
        "baseline_summary": _fmt_report(baseline_report),

        # BERTimbau
        "bertimbau_classification_report": bertimbau_report,
        "bertimbau_summary": _fmt_report(bertimbau_report),

        # Delivery & category cross-tabs
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
    if bertimbau_report is not None:
        print(f"    - bertimbau_sentiment/  (fine-tuned model + tokenizer)")

    # Print benchmark comparison
    if bertimbau_report is not None:
        b = _fmt_report(baseline_report)
        bert = _fmt_report(bertimbau_report)
        print("\n" + "=" * 70)
        print("MODEL BENCHMARK COMPARISON")
        print("=" * 70)
        metrics = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_f1"]
        header = f"{'Metric':<25} {'TF-IDF + LR':>15} {'BERTimbau':>15} {'Delta':>10}"
        print(header)
        print("-" * 70)
        for m in metrics:
            bv = b[m]
            bev = bert[m]
            delta = bev - bv
            sign = "+" if delta >= 0 else ""
            print(f"{m:<25} {bv:>15.4f} {bev:>15.4f} {sign}{delta:>9.4f}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment models on Olist reviews.")
    parser.add_argument(
        "--skip-bertimbau",
        action="store_true",
        default=False,
        help="Skip BERTimbau fine-tuning (use if you only want the fast TF-IDF baseline).",
    )
    args = parser.parse_args()
    main(skip_bertimbau=args.skip_bertimbau)
