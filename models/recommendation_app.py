import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="E-Commerce Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set base path to the directory where this script resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "artifacts")


# Helper function to generate recommendations
def get_recommendations(
    product_id,
    model_type="content",
    top_n=5,
    product_catalog=None,
    nn_content=None,
    combined_features=None,
    nn_svd=None,
    latent_matrix=None,
):
    if (
        product_catalog is None
        or product_id not in product_catalog["product_id"].values
    ):
        return pd.DataFrame()

    # FIX 1: Retrieve position-based integer index for matrix slicing
    idx = product_catalog.index[product_catalog["product_id"] == product_id][0]

    if model_type == "content":
        model = nn_content
        feature_matrix = combined_features
        query_vec = feature_matrix[idx]
    else:  # svd
        model = nn_svd
        feature_matrix = latent_matrix
        query_vec = feature_matrix[idx]

    # FIX 2: Corrected Indentation Error for query vector reshaping & neighbor calculations
    if hasattr(query_vec, "reshape") and getattr(query_vec, "ndim", 1) == 1:
        query_vec = query_vec.reshape(1, -1)

    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n + 1)

    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]

    res = product_catalog.iloc[rec_indices][
        ["product_id", "product_category_name_english", "price", "review_score"]
    ].copy()

    # FIX 3: Check metric type to properly convert distance to a similarity score (0 to 1)
    metric = getattr(model, "metric", "minkowski")
    if metric == "cosine":
        similarity = 1 - rec_distances
    else:
        similarity = 1 / (1 + rec_distances)

    res["similarity_score"] = np.round(similarity, 4)
    return res


# Load all model artifacts directly from the models/ folder
@st.cache_resource
def load_all_artifacts():
    catalog = joblib.load(os.path.join(MODELS_DIR, "product_catalog.pkl"))
    nn_content = joblib.load(os.path.join(MODELS_DIR, "knn_content_model.pkl"))
    combined_features = joblib.load(
        os.path.join(MODELS_DIR, "combined_features.pkl")
    )
    nn_svd = joblib.load(os.path.join(MODELS_DIR, "knn_svd_model.pkl"))
    latent_matrix = joblib.load(os.path.join(MODELS_DIR, "latent_matrix.pkl"))
    return catalog, nn_content, combined_features, nn_svd, latent_matrix


# Load resources
try:
    product_catalog, nn_content, combined_features, nn_svd, latent_matrix = (
        load_all_artifacts()
    )
except Exception as e:
    st.error(f"Error loading model artifacts from '{MODELS_DIR}': {e}")
    st.stop()

# Header Section
st.title("🛍️ E-Commerce Recommendation Engine")
st.markdown(
    "Discover similar products using **Content-Based Filtering** or **Matrix Factorization (SVD)**."
)
st.divider()

# Sidebar Setup
st.sidebar.header("⚙️ Configuration")

product_id_map = {
    row[
        "product_id"
    ]: f"{(str(row['product_category_name_english']).replace('_', ' ').title())} - ${row['price']:.2f} (ID: {row['product_id'][:8]}...)"
    for _, row in product_catalog.iterrows()
}

selected_id = st.sidebar.selectbox(
    "Select a Product:",
    options=list(product_id_map.keys()),
    format_func=lambda pid: product_id_map[pid],
)

model_choice = st.sidebar.radio(
    "Recommendation Strategy:",
    options=["content", "svd"],
    format_func=lambda x: (
        "Enhanced Content-Based (TF-IDF + Features)"
        if x == "content"
        else "Matrix Factorization (TruncatedSVD)"
    ),
)

top_n = st.sidebar.slider(
    "Number of Recommendations:", min_value=1, max_value=10, value=5
)

# Overview of Selected Product
selected_product_info = product_catalog[
    product_catalog["product_id"] == selected_id
].iloc[0]

st.subheader("📌 Selected Product Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Product ID",
        value=str(selected_product_info["product_id"])[:12] + "...",
    )
with col2:
    category_display = (
        str(selected_product_info["product_category_name_english"])
        .replace("_", " ")
        .replace("unknown", "General Items")
        .replace("Unknown", "General Items")
        .title()
    )
    st.metric(
        label="Category",
        value=category_display if category_display != "None" else "General Items",
    )
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
            latent_matrix=latent_matrix,
        )

        st.subheader(
            f"✨ Top {top_n} Recommendations ({'Content-Based' if model_choice == 'content' else 'TruncatedSVD'} Model)"
        )

        formatted_recs = recs.copy()
        if "price" in formatted_recs.columns:
            formatted_recs["price"] = formatted_recs["price"].map(
                "${:.2f}".format
            )
        if "product_category_name_english" in formatted_recs.columns:
            formatted_recs["product_category_name_english"] = (
                formatted_recs["product_category_name_english"]
                .str.replace("_", " ")
                .str.title()
            )

        st.dataframe(formatted_recs, use_container_width=True, hide_index=True)
else:
    st.info(
        "👈 Select a product and click **'Get Recommendations'** to view results."
    )