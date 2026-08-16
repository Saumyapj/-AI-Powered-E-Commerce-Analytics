"""
Generates granular Product Category x Customer State business analytics
for the interactive Drill-Down Explorer tab in the Streamlit app.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_pipeline import load_and_clean

DATA_PATH = ROOT / "data" / "olist_master_dataset.csv"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_product_state_analytics():
    print("=" * 60)
    print("GENERATING PRODUCT & STATE DRILL-DOWN ARTIFACTS")
    print("=" * 60)

    print("[1] Loading and cleaning dataset...")
    df = load_and_clean(str(DATA_PATH))
    df["year_month"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m")

    # 1. Product x State Summary
    print("[2] Aggregating Category x State metrics...")
    grouped = df.groupby(["product_category_name_english", "customer_state"])

    records = []
    for (cat, state), group in grouped:
        total_rev = group["payment_value"].sum()
        total_orders = group["order_id"].nunique()
        total_items = len(group)
        avg_price = group["price"].mean()
        avg_freight = group["freight_value"].mean()
        avg_freight_pct = group["freight_pct"].mean()

        # Delivery metrics
        delivered = group[group["order_delivered_customer_date"].notnull()]
        deliv_success_rate = (len(delivered) / len(group) * 100) if len(group) > 0 else 0
        late_deliv_rate = (len(group[group["delivery_status"] == "Late"]) / len(group) * 100) if len(group) > 0 else 0
        avg_deliv_days = delivered["delivery_days"].mean() if len(delivered) > 0 else 0

        # Sentiment metrics
        labeled_reviews = group[group["sentiment"].isin(["Positive", "Negative", "Neutral"])]
        n_reviews = len(labeled_reviews)
        if n_reviews > 0:
            pos_pct = (labeled_reviews["sentiment"] == "Positive").mean() * 100
            neg_pct = (labeled_reviews["sentiment"] == "Negative").mean() * 100
            neu_pct = (labeled_reviews["sentiment"] == "Neutral").mean() * 100
        else:
            pos_pct, neg_pct, neu_pct = 75.0, 15.0, 10.0

        returning_pct = group["is_returning_customer"].mean() * 100

        records.append({
            "category": cat,
            "state": state,
            "total_revenue": round(total_rev, 2),
            "total_orders": int(total_orders),
            "total_items": int(total_items),
            "avg_price": round(avg_price, 2),
            "avg_freight": round(avg_freight, 2),
            "freight_pct": round(avg_freight_pct, 1),
            "delivery_success_rate": round(deliv_success_rate, 1),
            "late_delivery_rate": round(late_deliv_rate, 1),
            "avg_delivery_days": round(avg_deliv_days, 1) if not pd.isna(avg_deliv_days) else 0.0,
            "positive_sentiment_pct": round(pos_pct, 1),
            "negative_sentiment_pct": round(neg_pct, 1),
            "neutral_sentiment_pct": round(neu_pct, 1),
            "total_reviews": int(n_reviews),
            "returning_customer_pct": round(returning_pct, 1),
        })

    summary_df = pd.DataFrame(records)
    summary_path = ARTIFACTS_DIR / "category_state_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"    Saved: {summary_path} ({len(summary_df)} rows)")

    # 2. Category x State Monthly Trend
    print("[3] Aggregating monthly trends...")
    monthly = df.groupby(["product_category_name_english", "customer_state", "year_month"]).agg(
        revenue=("payment_value", "sum"),
        orders=("order_id", "nunique"),
        avg_sentiment=("review_score", "mean")
    ).reset_index().rename(columns={
        "product_category_name_english": "category",
        "customer_state": "state"
    })
    monthly_path = ARTIFACTS_DIR / "category_state_monthly.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"    Saved: {monthly_path} ({len(monthly)} rows)")

    # 3. Sample customer reviews
    print("[4] Extracting sample customer reviews...")
    sample_reviews = {}
    for (cat, state), group in df.groupby(["product_category_name_english", "customer_state"]):
        with_comm = group[group["has_comment"] == 1]
        if with_comm.empty:
            continue
        key = f"{cat}___{state}"
        pos_examples = with_comm[with_comm["sentiment"] == "Positive"]["review_text"].head(2).tolist()
        neg_examples = with_comm[with_comm["sentiment"] == "Negative"]["review_text"].head(2).tolist()
        sample_reviews[key] = {
            "positive": pos_examples,
            "negative": neg_examples
        }

    reviews_path = ARTIFACTS_DIR / "category_state_reviews.json"
    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(sample_reviews, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {reviews_path}")

    print("\n[5] All drill-down analytics generated successfully!")


if __name__ == "__main__":
    generate_product_state_analytics()
