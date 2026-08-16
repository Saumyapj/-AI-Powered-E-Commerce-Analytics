import joblib
import pandas as pd
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_pipeline import load_and_clean

DATA_PATH = ROOT / "data" / "olist_master_dataset.csv"
ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Load artifacts directly from the models subfolder
product_catalog = joblib.load(os.path.join(ARTIFACTS, "product_catalog.pkl"))
knn_content = joblib.load(os.path.join(ARTIFACTS, "knn_content_model.pkl"))
combined_features = joblib.load(os.path.join(ARTIFACTS, "combined_features.pkl"))
knn_svd = joblib.load(os.path.join(ARTIFACTS, "knn_svd_model.pkl"))
latent_matrix = joblib.load(os.path.join(ARTIFACTS, "latent_matrix.pkl"))

def get_recommendations(product_id: str, model_type: str = "content", top_n: int = 5):
    """
    Returns top_n recommended products for a given product_id.
    model_type options: 'content' or 'svd'
    """
    if product_id not in product_catalog['product_id'].values:
        return None
    
    idx = product_catalog[product_catalog['product_id'] == product_id].index[0]
    
    if model_type == "svd":
        model = knn_svd
        matrix = latent_matrix
    else:
        model = knn_content
        matrix = combined_features
    
    distances, indices = model.kneighbors(matrix[idx], n_neighbors=top_n + 1)
    rec_indices = indices[0][1:]
    
    recs = product_catalog.iloc[rec_indices][
        ['product_id', 'product_category_name_english', 'price', 'review_score']
    ].copy()
    
    recs['similarity_score'] = [round(float(1 - d), 4) for d in distances[0][1:]]
    return recs

# Quick local test
if __name__ == "__main__":
    sample_id = product_catalog['product_id'].iloc[0]
    print(f"Testing recommendations for Product: {sample_id}\n")
    print(get_recommendations(sample_id, model_type="content"))