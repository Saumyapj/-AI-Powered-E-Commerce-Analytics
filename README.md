# 📊 AI-Powered E-Commerce Analytics Platform

An end-to-end analytics and AI dashboard for e-commerce data (built around the Olist Brazilian e-commerce dataset), combining BI dashboards, ML-driven forecasting/sentiment classification, customer satisfaction, recommendation engine, AI insights and a Retrieval-Augmented Generation (RAG) chatbot — all in a single Streamlit app.

---

## ✨ Features

| Tab | Description |
|---|---|
| 📊 **Dashboard** | Platform-wide KPIs (revenue, orders, AOV, sentiment, delivery success/late rate, cancellations, returning customers) plus revenue, category, state, and payment-method charts. |
| 🔎 **Product & State Drill-Down** | Filter any product category × customer state combination to see revenue, satisfaction, delivery performance, monthly trends, and real customer review excerpts for that slice. |
| 📈 **Sales Forecast** | 7–90 day GMV forecasting (XGBoost / RandomForest / Hybrid), scaled to any category/state slice, with WAPE, SMAPE, MAE, RMSE, and R² evaluation metrics. |
| 💬 **Customer Sentiment** | Sentiment distribution (Positive/Neutral/Negative), sentiment by delivery status and top category, and a live classifier you can test with your own review text (Portuguese). |
| 😊 **Customer Satisfaction Predictor** | Predicts a 1–5 star review score from order, product, delivery, and payment inputs using Random Forest, Decision Tree, and XGBoost classifiers, with a "best model" auto-selector. |
| 🎯 **Product Recommendations** | Content-based (KNN over combined product features) and collaborative-filtering (SVD latent-factor + KNN) recommenders. |
| 🤖 **AI Insights** | Auto-generated executive report (revenue, sentiment, logistics, growth, satisfaction drivers, recommendation strategy) templated from the underlying KPI/sentiment/forecast/reviews/logistics/products data. |
| 💡 **Ask Your Data** | English-language RAG chatbot — retrieves relevant chunks from a business knowledge base via **Pinecone** and generates answers with **Google Gemini**, with an automatic keyword-based fallback when no API keys / index are available. |

---

## 🗂️ Project Structure

```
project_root/
|
├── frontend/
│   └── streamlit_app.py           # Main Streamlit app 
├── models/
│   └── artifacts/                 # Trained models & precomputed data 
│       └── knowledge/*.txt        # Source docs for the RAG knowledge base
|           bertimbau_sentiment    # sentiment analysis
├── utils/
│   └── rag_engine.py              # RAGIndex, answer_question, extractive_fallback, NoRetrievedContext
|       build_index.py             # Builds/populates the Pinecone index from models/artifacts/
|       train_forecasting.py       # Forecasting
|       train_sentiment.py         # Sentiment analysis
|       recommendation_app.py      # Product Recommendation
|       customer_satisfaction_app.py # customer review
├── static/
│   └── style.css                  # Optional extra CSS  for customer satisfaction
├                 
└── README.md
```

### Required artifacts (`models/artifacts/`)

The app expects these files to already exist (produced by your training/preprocessing scripts):

**Data & KPIs**
- `kpi_summary.json`
- `sentiment_metadata.json`
- `forecasting_metadata.json`
- `daily_gmv_history.csv`
- `future_forecast_90d.csv`
- `category_state_summary.csv`
- `category_state_monthly.csv`
- `category_state_reviews.json`

**Sentiment classifier**
 `tfidf_vectorizer.pkl`
- `sentiment_model.pkl`
- `sentiment_metadata.json`
- `model_safetensors`
- `tokenizer.json`

**Product recommendations**
- `product_catalog.pkl`
- `knn_content_model.pkl`
- `combined_features.pkl`
- `knn_svd_model.pkl`
- `latent_matrix.pkl`

**Customer satisfaction prediction**
- `best_sentiment_model.pkl`
- `random_forest_best.pkl`
- `decision_tree_best.pkl`
- `xgboost_best.pkl`
- `scaler.pkl`
- `label_encoders.pkl`

**AI Chatbot**
(`models/artifacts/knowledge`)
- 'business_insights.txt'
- 'customer_analysis.txt'
- 'dataset_overview.txt'
- 'delivery_analysis.txt'
- 'forecasting_results.txt'
- 'product_analysis.txt'
- 'recommendation_results.txt'
- 'sentiment_analysis.txt'

If any file is missing, the app will show an error pointing at the exact missing path — run the corresponding training/export script in `models/` first.

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn joblib xgboost openpyxl
Pip install torch transformers python-dotenv langchain-community langchain-text-splitters langchain-google-genai langchain-pinecone langchain-core pinecone sentence-transformers torchvision
```
versions
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.15.0
scikit-learn>=1.2.0
joblib>=1.2.0
xgboost>=1.7.0
openpyxl>=3.1.0
torch>=2.0.0
transformers>=4.30.0

### 2. Configure secrets
The chatbot tab (**Ask Your Data**) needs:

| Variable | Purpose |
|---|---|
| `PINECONE_API_KEY` | Retrieval over your business knowledge base |
| `GOOGLE_API_KEY` | Answer generation via Gemini |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index to query |

**Local development** — export them in your shell, or use a `.env` file:
```bash
export PINECONE_API_KEY="..."
export GOOGLE_API_KEY="..."
export PINECONE_INDEX_NAME="..."
```

**Streamlit Community Cloud** — set them under your app's **Settings → Secrets**, or locally in `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "..."
GOOGLE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
```

Without these, the chatbot tab still works — it falls back to a lightweight keyword-based Q&A over the dashboard KPIs instead of RAG.

### 3. Build the RAG index (optional, for the chatbot)

```bash
python build_index.py
```
This reads `models/artifacts/knowledge/*.txt` and populates the Pinecone index used by the **Ask Your Data** tab.

### 4. Run the app

```bash
streamlit run frontend/streamlit_app.py
```

---

## 🧠 How each ML piece works

- **Sentiment classifier** — Portuguese text is cleaned (lowercased, HTML/URLs/punctuation/digits stripped, stopwords removed) and vectorized with a fitted TF-IDF vectorizer, then classified as Positive/Neutral/Negative. If the model/vectorizer fails to load or predict, the app reloads them fresh from disk, and as a last resort falls back to a small Portuguese keyword heuristic.
- **Sales forecasting** — Precomputed daily GMV history and a 90-day forecast are combined and scaled by the revenue share of whatever category/state slice is selected, so the same model output can be viewed at the platform level or drilled into a specific segment.
- **Customer satisfaction prediction** — Order, product, payment, and delivery features are label-encoded and scaled, then passed to Random Forest / Decision Tree / XGBoost classifiers (trained with SMOTE to handle class imbalance) to predict a 1–5 star review score.
- **Recommendations** — Content-based filtering uses KNN over combined product feature vectors; collaborative filtering uses KNN over an SVD latent-factor matrix. Both return the nearest products with a similarity score.
- **AI Insights report** — Not an LLM call; it's a templated report assembled in Python directly from the KPI, sentiment, customer satisfaction, product recommendation and forecast JSON artifacts, with a focus-area selector to show a subset of sections and give recommendation strategy to improve the same.
- **Ask Your Data chatbot** — Retrieves relevant chunks from Pinecone, generates a grounded answer with Gemini, and shows source excerpts on request. Falls back to `answer_chat()`, a keyword-matching assistant over the KPI/sentiment/forecast/recommendation/customer satisfaction data, when RAG isn't available or doesn't retrieve anything relevant.

---

## 🛠️ Tech Stack

- **UI:** Streamlit, custom dark theme (CSS)
- **Visualization:** Plotly
- **ML:** scikit-learn (TF-IDF, BERTimbau, Random Forest, Decision Tree, KNN), XGBoost, joblib for model persistence
- **RAG:** Pinecone (vector retrieval) + Google Gemini (generation)
- **Data:** pandas, numpy

---

## 📝 Notes

- Currency values are formatted in R$ (Brazilian Real), consistent with the Olist dataset.
