
import numpy as np
import pandas as pd

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw Olist master dataset and apply the shared cleaning /
    feature-engineering steps used by both the sales-forecasting and
    sentiment-analysis notebooks."""

    df = pd.read_csv(csv_path)

    # ---- Parse dates -----------------------------------------------------
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # ---- Calendar features -------------------------------------------
    df["year"] = df["order_purchase_timestamp"].dt.year
    df["month"] = df["order_purchase_timestamp"].dt.month
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["dayofyear"] = df["order_purchase_timestamp"].dt.dayofyear
    df["hour"] = df["order_purchase_timestamp"].dt.hour
    df["week"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # ---- Delivery features -------------------------------------------
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["delivery_delay"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["carrier_pickup_days"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.days
    df["transit_days"] = (
        df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
    ).dt.days

    df["delivery_status"] = np.where(df["delivery_delay"] > 0, "Late", "On Time")
    df.loc[df["order_delivered_customer_date"].isnull(), "delivery_status"] = "Not Delivered Yet"

    # ---- Customer type (fixed vs. notebook bug) -----------------------
    order_counts = df.groupby("customer_unique_id")["order_id"].transform("nunique")
    df["Customer_Type"] = np.where(order_counts > 1, "Returning", "New")
    df["is_returning_customer"] = (df["Customer_Type"] == "Returning").astype(int)

    # ---- Payment features ----------------------------------------------
    df["payment_methods_per_order"] = df.groupby("order_id")["payment_sequential"].transform("max")
    df["is_multi_payment"] = (df["payment_methods_per_order"] > 1).astype(int)
    df["is_installment"] = (df["payment_installments"] > 1).astype(int)

    # ---- Review / sentiment features -----------------------------------
    df["review_comment_title"] = df["review_comment_title"].fillna("No Title")
    df["review_comment_message"] = df["review_comment_message"].fillna("No Review")
    df["review_text"] = (
        df["review_comment_title"].replace("No Title", "") + " " +
        df["review_comment_message"].replace("No Review", "")
    ).str.strip()
    df["has_comment"] = (df["review_text"] != "").astype(int)
    df["comment_length"] = df["review_text"].str.len()

    def _sentiment(score):
        if pd.isna(score):
            return "Unknown"
        if score <= 2:
            return "Negative"
        if score == 3:
            return "Neutral"
        return "Positive"

    df["sentiment"] = df["review_score"].apply(_sentiment)

    # ---- Missing value handling -----------------------------------------
    df["payment_value"] = df["payment_value"].fillna(df["payment_value"].median())
    df["product_category_name_english"] = df["product_category_name_english"].fillna("Unknown")
    df["freight_value"] = df.groupby(["customer_state"])["freight_value"].transform(
        lambda x: x.fillna(x.median())
    )
    df["freight_value"] = df["freight_value"].fillna(df["freight_value"].median())
    df["order_item_id"] = df["order_item_id"].fillna(1)
    df["price"] = df.groupby("product_category_name_english")["price"].transform(
        lambda x: x.fillna(x.median())
    )
    df["price"] = df["price"].fillna(df["price"].median())
    median_installments = df["payment_installments"].median()
    df["payment_installments"] = df["payment_installments"].fillna(median_installments)

    # ---- Derived monetary features --------------------------------------
    df["total_item_value"] = df["price"] + df["freight_value"]
    df["freight_pct"] = (df["freight_value"] / df["price"].replace(0, np.nan)) * 100
    df["freight_pct"] = df["freight_pct"].replace([np.inf, -np.inf], 0).fillna(0)

    # Cap freight_pct outliers (IQR)
    q1, q3 = df["freight_pct"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["freight_pct"] = df["freight_pct"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/olist_master_dataset.csv"
    df = load_and_clean(path)
    print(f"Loaded & cleaned dataset: {df.shape}")
    print(df[["order_purchase_timestamp", "payment_value", "sentiment", "delivery_status"]].head())
