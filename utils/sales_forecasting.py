"""
Sales / revenue forecasting engine.

Reuses and cleans up the SalesFeatureEncoder / TSAggregator / TSFeatureEngineer /
HybridForecaster design from Capstone_Project_E_commerce_sales_forecasting.ipynb.
Only sklearn is required (no XGBoost/LightGBM/Prophet/TensorFlow dependency) —
RandomForest and the HistGradientBoosting+Ridge Hybrid were the notebook's
top-performing models anyway.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SalesFeatureEncoder:
    """Fourier seasonality encoding + Pareto top-10 category / top-3 state one-hot."""

    TOP_10_CATEGORIES = [
        "bed_bath_table", "health_beauty", "sports_leisure", "watches_gifts",
        "computers_accessories", "furniture_decor", "housewares", "telephony",
        "auto", "garden_tools",
    ]
    TOP_3_STATES = ["SP", "RJ", "MG"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
        df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52.0)
        df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52.0)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["sin_dayofyear"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
        df["cos_dayofyear"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)

        for cat in self.TOP_10_CATEGORIES:
            df[f"cat_{cat}"] = (df["product_category_name_english"] == cat).astype(int)
        df["cat_other"] = (~df["product_category_name_english"].isin(self.TOP_10_CATEGORIES)).astype(int)

        for state in self.TOP_3_STATES:
            df[f"state_{state}"] = (df["customer_state"] == state).astype(int)
        df["state_other"] = (~df["customer_state"].isin(self.TOP_3_STATES)).astype(int)

        return df


class TSAggregator:
    """Collapse order-line-level rows into a daily GMV time series panel."""

    @staticmethod
    def aggregate_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
        valid = df[df["order_status"] == "delivered"].copy()
        valid["date"] = valid["order_purchase_timestamp"].dt.date

        agg_rules = {
            "payment_value": "sum",
            "total_item_value": "sum",
            "freight_value": "sum",
            "order_item_id": "count",
            "price": "mean",
            "freight_pct": "mean",
            "payment_installments": "mean",
            "is_installment": "mean",
            "sin_dayofweek": "first", "cos_dayofweek": "first",
            "sin_week": "first", "cos_week": "first",
            "sin_month": "first", "cos_month": "first",
            "sin_dayofyear": "first", "cos_dayofyear": "first",
            "year": "first", "quarter": "first", "month": "first",
            "is_weekend": "first",
            "is_returning_customer": "mean",
        }
        for col in valid.columns:
            if col.startswith("cat_") or col.startswith("state_"):
                agg_rules[col] = "mean"

        daily = valid.groupby("date", as_index=False).agg(agg_rules)
        daily = daily.rename(columns={
            "payment_value": "daily_gmv",
            "total_item_value": "daily_item_revenue",
            "freight_value": "shipping_revenue",
            "order_item_id": "order_volume",
            "price": "avg_price",
            "freight_pct": "avg_freight_pct",
            "payment_installments": "avg_installments",
            "is_installment": "is_installment_rate",
        })
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        daily["is_black_friday"] = ((daily["month"] == 11) & (daily["date"].dt.day >= 22)).astype(int)
        daily["is_holiday_season"] = ((daily["month"] == 12) & (daily["date"].dt.day <= 24)).astype(int)
        return daily


class TSFeatureEngineer:
    """Autoregressive lag & rolling-window features."""

    @staticmethod
    def engineer_features(daily_ts: pd.DataFrame, target_col="daily_gmv") -> pd.DataFrame:
        d = daily_ts.copy().sort_values("date").reset_index(drop=True)
        for lag in [1, 2, 3, 7, 14, 21, 30]:
            d[f"lag_{lag}"] = d[target_col].shift(lag)
        for w in [7, 14, 30]:
            d[f"roll_mean_{w}"] = d[target_col].shift(1).rolling(window=w).mean()
            d[f"roll_std_{w}"] = d[target_col].shift(1).rolling(window=w).std()
        return d.dropna().reset_index(drop=True)


class HybridForecaster:
    """60% HistGradientBoosting (non-linear, lag-aware) + 40% Fourier Ridge (seasonality)."""

    FEATURE_COLS = [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_30",
        "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_mean_30",
        "is_weekend", "sin_dayofweek", "cos_dayofweek", "sin_week", "cos_week",
        "sin_month", "cos_month", "sin_dayofyear", "cos_dayofyear",
        "is_black_friday", "is_holiday_season", "quarter", "year",
        "avg_freight_pct", "avg_installments", "avg_price",
    ]
    LINEAR_COLS = [
        "sin_dayofweek", "cos_dayofweek", "sin_week", "cos_week",
        "sin_month", "cos_month", "sin_dayofyear", "cos_dayofyear",
        "is_weekend", "is_black_friday", "is_holiday_season", "quarter", "year",
        "avg_freight_pct", "avg_installments", "avg_price",
    ]

    def __init__(self, tree_weight=0.6, linear_weight=0.4):
        self.tree_weight = tree_weight
        self.linear_weight = linear_weight
        self.tree_model = HistGradientBoostingRegressor(
            max_iter=150, max_depth=6, learning_rate=0.05, random_state=42
        )
        self.linear_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.tree_model.fit(X[self.FEATURE_COLS], y)
        self.linear_model.fit(X[self.LINEAR_COLS], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_tree = self.tree_model.predict(X[self.FEATURE_COLS])
        pred_linear = self.linear_model.predict(X[self.LINEAR_COLS])
        return np.maximum(self.tree_weight * pred_tree + self.linear_weight * pred_linear, 500.0)


def evaluate(y_true, y_pred, model_name="model"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "WAPE_%": round(wape, 2), "SMAPE_%": round(smape, 2),
            "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def build_daily_panel(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: encode -> aggregate -> lag/rolling features."""
    encoded = SalesFeatureEncoder().transform(df_clean)
    daily = TSAggregator.aggregate_daily_panel(encoded)
    features = TSFeatureEngineer.engineer_features(daily, target_col="daily_gmv")
    return features
