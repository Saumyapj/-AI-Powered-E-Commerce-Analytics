import os
import streamlit as st
import pandas as pd
import joblib

# Streamlit Page Configuration
st.set_page_config(
    page_title="E-Commerce Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set base path to the directory where this script resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Helper function to generate recommendations
def get_recommendations(product_id, model_type="content", top_n=5, product_catalog=None, nn_content=None, combined_features=None, nn_svd=None, latent_matrix=None):
    if product_id not in product_catalog['product_id'].values:
        return pd.DataFrame()

    idx = product_catalog[product_catalog['product_id'] == product_id].index[0]

    if model_type == "content":
        model = nn_content
        feature_matrix = combined_features
        query_vec = feature_matrix[idx]
    else: # svd
        model = nn_svd
        feature_matrix = latent_matrix
        query_vec = feature_matrix[idx].reshape(1, -1)

    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n+1)

    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]

    res = product_catalog.iloc[rec_indices][['product_id', 'product_category_name_english', 'price', 'review_score']].copy()
    res['similarity_score'] = (1 - rec_distances).round(4)
    return res

# Load all model artifacts directly from the models/ folder
@st.cache_resource
def load_all_artifacts():
    catalog = joblib.load(os.path.join(MODELS_DIR, 'product_catalog.pkl'))
    nn_content = joblib.load(os.path.join(MODELS_DIR, 'knn_content_model.pkl'))
    combined_features = joblib.load(os.path.join(MODELS_DIR, 'combined_features.pkl'))
    nn_svd = joblib.load(os.path.join(MODELS_DIR, 'knn_svd_model.pkl'))
    latent_matrix = joblib.load(os.path.join(MODELS_DIR, 'latent_matrix.pkl'))
    return catalog, nn_content, combined_features, nn_svd, latent_matrix

# Load resources
try:
    product_catalog, nn_content, combined_features, nn_svd, latent_matrix = load_all_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts from 'models/' directory: {e}")
    st.stop()

# Header Section
st.title("🛍️ E-Commerce Recommendation Engine")
st.markdown("Discover similar products using **Content-Based Filtering** or **Matrix Factorization (SVD)**.")
st.divider()

# Sidebar Setup
st.sidebar.header("⚙️ Configuration")

product_list = product_catalog['product_id'].tolist()
selected_id = st.sidebar.selectbox(
    "Select a Product ID:",
    options=product_list
)

model_choice = st.sidebar.radio(
    "Recommendation Strategy:",
    options=["content", "svd"],
    format_func=lambda x: "Enhanced Content-Based (TF-IDF + Features)" if x == "content" else "Matrix Factorization (TruncatedSVD)"
)

top_n = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

# Overview of Selected Product
selected_product_info = product_catalog[product_catalog['product_id'] == selected_id].iloc[0]

st.subheader("📌 Selected Product Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Product ID", value=str(selected_product_info['product_id'])[:12] + "...")
with col2:
    st.metric(label="Category", value=str(selected_product_info['product_category_name_english']).replace("_", " ").title())
with col3:
    st.metric(label="Price", value=f"${selected_product_info['price']:.2f}")

st.divider()

# Action Button
if st.button("🚀 Get Recommendations", type="primary"):
    with st.spinner("Calculating recommendations..."):
        recs = get_recommendations(
            product_id=selected_id,
            model_type=model_choice,
            top_n=top_n,
            product_catalog=product_catalog,
            nn_content=nn_content,
            combined_features=combined_features,
            nn_svd=nn_svd,
            latent_matrix=latent_matrix
        )

        st.subheader(f"✨ Top {top_n} Recommendations ({'Content-Based' if model_choice == 'content' else 'TruncatedSVD'} Model)")

        formatted_recs = recs.copy()
        if 'price' in formatted_recs.columns:
            formatted_recs['price'] = formatted_recs['price'].map("${:.2f}".format)
        if 'product_category_name_english' in formatted_recs.columns:
            formatted_recs['product_category_name_english'] = formatted_recs['product_category_name_english'].str.replace("_", " ").str.title()

        st.dataframe(
            formatted_recs,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("👈 Select a product and click **'Get Recommendations'** to view results.")