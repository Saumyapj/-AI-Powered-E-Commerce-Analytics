import os
import joblib
import numpy as np
import pandas as pd


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
    if product_catalog is None or product_id not in product_catalog['product_id'].values:
        return pd.DataFrame()

    # Get integer position index regardless of custom index
    idx = product_catalog.index[product_catalog['product_id'] == product_id][0]

    if model_type == "content":
        model = nn_content
        feature_matrix = combined_features
        query_vec = feature_matrix[idx]
    else:  # SVD
        model = nn_svd
        feature_matrix = latent_matrix
        query_vec = (
            feature_matrix[idx].reshape(1, -1)
            if hasattr(feature_matrix[idx], 'reshape')
            else feature_matrix[idx]
        )

    # Ensure query_vec is 2D array
    if hasattr(query_vec, 'ndim') and query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n + 1)

    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]

    res = product_catalog.iloc[rec_indices][
        ['product_id', 'product_category_name_english', 'price', 'review_score']
    ].copy()

    # Normalize similarity score based on metric type
    metric = getattr(model, 'metric', 'minkowski')
    if metric == 'cosine':
        similarity = 1 - rec_distances
    else:
        similarity = 1 / (1 + rec_distances)

    res['similarity_score'] = np.round(similarity, 4)
    return res