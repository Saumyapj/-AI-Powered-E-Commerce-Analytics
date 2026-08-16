"""
Trains the sales-forecasting models (RandomForest + HybridForecaster) on the
Olist dataset and pickles them, along with evaluation metrics and 90-day
future forecasts, for the backend to serve.

Run from the project root:
    python models/train_forecasting.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, LGBMRegressor 

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_pipeline import load_and_clean
from utils.sales_forecasting import (
    HybridForecaster, SalesFeatureEncoder, TSAggregator, TSFeatureEngineer, evaluate,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "olist_master_dataset.csv"
MODEL_DIR = ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("SALES & REVENUE FORECASTING — TRAINING PIPELINE")
    print("=" * 70)

    print("[1] Loading & cleaning raw data...")
    df = load_and_clean(str(DATA_PATH))
    print(f"    Cleaned dataset: {df.shape}")

    print("[2] Encoding seasonal + category/state features...")
    encoded = SalesFeatureEncoder().transform(df)

    print("[3] Aggregating into daily GMV panel...")
    daily_ts = TSAggregator.aggregate_daily_panel(encoded)
    print(f"    Daily panel shape: {daily_ts.shape} "
          f"({daily_ts['date'].min().date()} to {daily_ts['date'].max().date()})")

    print("[4] Engineering lag & rolling-window features...")
    df_features = TSFeatureEngineer.engineer_features(daily_ts, target_col="daily_gmv")
    print(f"    Supervised matrix shape: {df_features.shape}")

    test_days = 30
    train_df = df_features.iloc[:-test_days].copy()
    test_df = df_features.iloc[-test_days:].copy()

    # NOTE: restrict to features that are genuinely knowable ahead of time
    # (lags/rolling stats of the target + calendar signals). Same-day
    # aggregates like order_volume, shipping_revenue, or category/state
    # mix-shares are themselves outputs of that day's orders and would leak
    # information a real forecast wouldn't have in advance.
    feature_cols = HybridForecaster.FEATURE_COLS

    X_train, y_train = train_df[feature_cols], train_df["daily_gmv"]
    X_test, y_test = test_df[feature_cols], test_df["daily_gmv"]

    results = []
    trained_models = {}

    print("\n[5] Training RandomForest...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                random_state=42, n_jobs=-1, bootstrap=True)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append(evaluate(y_test, rf_pred, "RandomForest"))
    trained_models["RandomForest"] = rf
    print(f"    done in {time.time()-t0:.1f}s -> {results[-1]}")

    print("\n[5] Training LGBM...")
    t0 = time.time()
    lgbm_model = LGBMRegressor(n_estimators=500,learning_rate=0.05,force_row_wise='true',
                                random_state=42  )
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    results.append(evaluate(y_test, lgbm_pred, "LGBM"))
    trained_models["LGBM"] = lgbm_model
    print(f"    done in {time.time()-t0:.1f}s -> {results[-1]}")

    """print("\n[6] Training HybridForecaster (60% HistGB + 40% Fourier Ridge)...")
    t0 = time.time()
    hybrid = HybridForecaster(tree_weight=0.6, linear_weight=0.4)
    hybrid.fit(train_df, y_train)
    hybrid_pred = hybrid.predict(test_df)
    results.append(evaluate(y_test, hybrid_pred, "Hybrid"))
    trained_models["Hybrid"] = hybrid
    print(f"    done in {time.time()-t0:.1f}s -> {results[-1]}")"""

    eval_df = pd.DataFrame(results).sort_values("WAPE_%")
    print("\n[7] Model comparison (lower WAPE/SMAPE/RMSE, higher R2 = better):")
    print(eval_df.to_string(index=False))

    best_model_name = eval_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n    Best model: {best_model_name}")

    # ---- Retrain best model on FULL data & forecast next 90 days ---------
    print(f"\n[8] Retraining {best_model_name} on full history & forecasting 90 days ahead...")
    full_X, full_y = df_features[feature_cols], df_features["daily_gmv"]
    if best_model_name == "RandomForest":
        final_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                             random_state=42, n_jobs=-1, bootstrap=True)
        final_model.fit(full_X, full_y)
    else:
        final_model = LGBMRegressor(n_estimators=500,learning_rate=0.05,force_row_wise='true',
                                random_state=42  )
        final_model.fit(df_features, full_y)

    future_forecast = _forecast_future(daily_ts, final_model, best_model_name, feature_cols, horizon=90)

    # ---- Persist artifacts -------------------------------------------------
    joblib.dump(trained_models, MODEL_DIR / "trained_forecasting_models.pkl")
    joblib.dump(final_model, MODEL_DIR / "best_forecasting_model.pkl")
    daily_ts.to_csv(MODEL_DIR / "daily_gmv_history.csv", index=False)
    future_forecast.to_csv(MODEL_DIR / "future_forecast_90d.csv", index=False)

    metadata = {
        "best_model": best_model_name,
        "trained_at": pd.Timestamp.now().isoformat(),
        "history_range": [str(daily_ts["date"].min().date()), str(daily_ts["date"].max().date())],
        "evaluation": results,
        "feature_cols": feature_cols,
    }
    with open(MODEL_DIR / "forecasting_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[9] Saved artifacts to {MODEL_DIR}/")
    print("    - trained_forecasting_models.pkl (all models)")
    print("    - best_forecasting_model.pkl")
    print("    - daily_gmv_history.csv")
    print("    - future_forecast_90d.csv")
    print("    - forecasting_metadata.json")
    print("\nDone.")


def _forecast_future(daily_ts, model, model_name, feature_cols, horizon=90):
    """Iteratively roll forward day-by-day, regenerating lag/rolling features
    from prior (actual + predicted) daily_gmv values."""
    history = daily_ts[["date", "daily_gmv"]].copy()
    last_date = history["date"].max()

    # carry forward the last known values of slow-moving exogenous columns
    last_row = daily_ts.iloc[-1]
    exogenous_defaults = {
        "avg_freight_pct": daily_ts["avg_freight_pct"].tail(30).mean(),
        "avg_installments": daily_ts["avg_installments"].tail(30).mean(),
        "avg_price": daily_ts["avg_price"].tail(30).mean(),
    }

    future_rows = []
    extended = history.copy()

    for i in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=i)
        row = {
            "date": d,
            "dayofweek": d.dayofweek, "week": d.isocalendar().week,
            "month": d.month, "dayofyear": d.dayofyear,
            "quarter": (d.month - 1) // 3 + 1, "year": d.year,
            "is_weekend": int(d.dayofweek in [5, 6]),
            "is_black_friday": int(d.month == 11 and d.day >= 22),
            "is_holiday_season": int(d.month == 12 and d.day <= 24),
        }
        row["sin_dayofweek"] = np.sin(2 * np.pi * row["dayofweek"] / 7.0)
        row["cos_dayofweek"] = np.cos(2 * np.pi * row["dayofweek"] / 7.0)
        row["sin_week"] = np.sin(2 * np.pi * row["week"] / 52.0)
        row["cos_week"] = np.cos(2 * np.pi * row["week"] / 52.0)
        row["sin_month"] = np.sin(2 * np.pi * row["month"] / 12.0)
        row["cos_month"] = np.cos(2 * np.pi * row["month"] / 12.0)
        row["sin_dayofyear"] = np.sin(2 * np.pi * row["dayofyear"] / 365.0)
        row["cos_dayofyear"] = np.cos(2 * np.pi * row["dayofyear"] / 365.0)
        row.update(exogenous_defaults)

        for lag in [1, 2, 3, 7, 14, 21, 30]:
            row[f"lag_{lag}"] = extended["daily_gmv"].iloc[-lag]
        for w in [7, 14, 30]:
            row[f"roll_mean_{w}"] = extended["daily_gmv"].tail(w).mean()
            row[f"roll_std_{w}"] = extended["daily_gmv"].tail(w).std()

        X_row = pd.DataFrame([row])[feature_cols]
        if model_name == "RandomForest":
            pred = float(model.predict(X_row)[0])
        else:
            pred = float(model.predict(pd.DataFrame([row]))[0])

        row["daily_gmv_forecast"] = pred
        future_rows.append(row)
        extended = pd.concat([extended, pd.DataFrame([{"date": d, "daily_gmv": pred}])], ignore_index=True)

    future_df = pd.DataFrame(future_rows)[["date", "daily_gmv_forecast", "is_black_friday", "is_holiday_season"]]
    return future_df


if __name__ == "__main__":
    main()
