"""
Computes headline business KPIs and dashboard-ready aggregates from the
cleaned dataset. Saves to models/artifacts/kpi_summary.json — this is what
powers the dashboard cards, charts, and grounds the AI insights/chat layer.

Run from the project root:
    python models/generate_kpis.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_pipeline import load_and_clean

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "olist_master_dataset.csv"
MODEL_DIR = ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = load_and_clean(str(DATA_PATH))

    total_revenue = float(df["payment_value"].sum())
    total_orders = int(df["order_id"].nunique())
    total_customers = int(df["customer_unique_id"].nunique())
    total_sellers = int(df["seller_id"].nunique())
    avg_order_value = total_revenue / total_orders
    delivery_success_rate = float((df["order_status"] == "delivered").mean() * 100)
    cancellation_rate = float((df["order_status"] == "canceled").mean() * 100)
    avg_review_score = float(df["review_score"].mean())

    delivered = df[df["order_status"] == "delivered"]
    late_rate = float((delivered["delivery_status"] == "Late").mean() * 100)
    avg_delivery_days = float(delivered["delivery_days"].mean())

    top_states = df["customer_state"].value_counts().head(10).to_dict()
    top_categories_revenue = (
        df.groupby("product_category_name_english")["payment_value"].sum()
        .sort_values(ascending=False).head(10).round(2).to_dict()
    )
    payment_method_share = (
        df["payment_type"].value_counts(normalize=True).mul(100).round(2).to_dict()
    )

    # revenue by month for a simple trend chart
    monthly = (
        df[df["order_status"] == "delivered"]
        .assign(month=lambda d: d["order_purchase_timestamp"].dt.to_period("M").astype(str))
        .groupby("month")["payment_value"].sum().round(2)
    )
    revenue_by_month = monthly.to_dict()

    returning_rate = float(df["is_returning_customer"].mean() * 100)

    kpis = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_range": [
            str(df["order_purchase_timestamp"].min().date()),
            str(df["order_purchase_timestamp"].max().date()),
        ],
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "total_sellers": total_sellers,
        "avg_order_value": round(avg_order_value, 2),
        "delivery_success_rate_pct": round(delivery_success_rate, 2),
        "cancellation_rate_pct": round(cancellation_rate, 2),
        "late_delivery_rate_pct": round(late_rate, 2),
        "avg_delivery_days": round(avg_delivery_days, 2),
        "avg_review_score": round(avg_review_score, 2),
        "returning_customer_rate_pct": round(returning_rate, 2),
        "top_states_by_customers": top_states,
        "top_categories_by_revenue": top_categories_revenue,
        "payment_method_share_pct": payment_method_share,
        "revenue_by_month": revenue_by_month,
    }

    with open(MODEL_DIR / "kpi_summary.json", "w") as f:
        json.dump(kpis, f, indent=2)

    print(json.dumps(kpis, indent=2))
    print(f"\nSaved to {MODEL_DIR / 'kpi_summary.json'}")


if __name__ == "__main__":
    main()
