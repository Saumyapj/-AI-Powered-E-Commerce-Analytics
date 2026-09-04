import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Page Config ---
st.set_page_config(
    page_title="Customer Sentiment Predictor",
    page_icon="🛒",
    layout="wide"
)

# --- Load Pickled Models ---
@st.cache_resource
def load_models():
    with open('models/best_sentiment_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('models/random_forest_best.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/decision_tree_best.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('models/xgboost_best.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return best_model, rf_model, dt_model, xgb_model, scaler, label_encoders

best_model, rf_model, dt_model, xgb_model, scaler, label_encoders = load_models()

# --- Title ---
st.title("🛒 Customer Sentiment Prediction")
st.markdown("**Olist Brazilian E-Commerce — Review Score Predictor (1-5)**")
st.markdown("---")

# ============================================================
# INPUT FORM - All 23 features exactly as training
# ============================================================

# --- EXACT FEATURE ORDER (must match training) ---
# 1. order_status          (categorical)
# 2. customer_zip_code_prefix (numerical)
# 3. customer_city         (categorical)
# 4. customer_state        (categorical)
# 5. order_item_id         (numerical)
# 6. price                 (numerical)
# 7. freight_value         (numerical)
# 8. product_category_name (categorical)
# 9. product_name_lenght   (numerical)
# 10. product_description_lenght (numerical)
# 11. product_photos_qty   (numerical)
# 12. product_weight_g     (numerical)
# 13. product_length_cm    (numerical)
# 14. product_height_cm    (numerical)
# 15. product_width_cm     (numerical)
# 16. payment_sequential   (numerical)
# 17. payment_type         (categorical)
# 18. payment_installments (numerical)
# 19. payment_value        (numerical)
# 20. product_category_name_english (categorical)
# 21. delivery_days        (numerical)
# 22. delivery_vs_estimated (numerical)
# 23. approval_hours       (numerical)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader(" Product Details")
    price = st.number_input("Price (R$)", min_value=0.0, value=120.50, step=1.0)
    freight_value = st.number_input("Freight Value (R$)", min_value=0.0, value=18.00, step=1.0)
    product_weight_g = st.number_input("Product Weight (g)", min_value=0, value=1500, step=50)
    product_length_cm = st.number_input("Length (cm)", min_value=1, value=30, step=1)
    product_height_cm = st.number_input("Height (cm)", min_value=1, value=10, step=1)
    product_width_cm = st.number_input("Width (cm)", min_value=1, value=20, step=1)
    product_photos_qty = st.number_input("Number of Photos", min_value=1, value=3, step=1)
    product_name_lenght = st.number_input("Product Name Length", min_value=1, value=50, step=1)
    product_description_lenght = st.number_input("Description Length", min_value=1, value=500, step=10)

with col2:
    st.subheader(" Payment & Order")
    payment_value = st.number_input("Payment Value (R$)", min_value=0.0, value=138.50, step=1.0)
    payment_installments = st.number_input("Installments", min_value=1, max_value=24, value=3, step=1)
    payment_sequential = st.number_input("Payment Sequential", min_value=1, max_value=5, value=1, step=1)
    order_item_id = st.number_input("Order Item ID", min_value=1, max_value=10, value=1, step=1)
    customer_zip_code_prefix = st.number_input("Customer Zip Code Prefix", min_value=1000, max_value=99999, value=1038, step=1)
    
    payment_type = st.selectbox("Payment Type", 
        options=['credit_card', 'boleto', 'voucher', 'debit_card'])
    
    order_status = st.selectbox("Order Status", 
        options=['delivered', 'shipped', 'invoiced'])
    
    product_category_name = st.selectbox("Product Category (Portuguese)", 
        options=['utilidades_domesticas', 'perfumaria', 'automotivo', 'pet_shop', 
                 'papelaria', 'brinquedos', 'moveis_decoracao', 'bebes',
                 'ferramentas_jardim', 'informatica_acessorios', 'esporte_lazer',
                 'cama_mesa_banho', 'beleza_saude', 'relogios_presentes'])

with col3:
    st.subheader(" Delivery & Location")
    delivery_days = st.number_input("Delivery Days", min_value=0, value=10, step=1)
    delivery_vs_estimated = st.number_input("Days vs Estimated (+ = early, - = late)", 
                                             min_value=-30, max_value=60, value=5, step=1)
    approval_hours = st.number_input("Approval Hours", min_value=0.0, value=2.5, step=0.5)
    
    customer_state = st.selectbox("Customer State", 
        options=['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'SC', 'DF', 
                 'GO', 'RN', 'PE', 'CE', 'PA', 'ES', 'MT', 'MA'])
    
    customer_city = st.text_input("Customer City", value="sao paulo")
    
    product_category_name_english = st.selectbox("Product Category (English)", 
        options=['housewares', 'perfumery', 'auto', 'pet_shop', 'stationery',
                 'toys', 'furniture_decor', 'baby', 'garden_tools',
                 'computers_accessories', 'sports_leisure', 'bed_bath_table',
                 'health_beauty', 'watches_gifts'])

# --- Model Selection ---
st.markdown("---")
st.subheader(" Model Selection")
model_choice = st.selectbox("Choose Prediction Model", 
    options=['Best Model (Auto)', 'Random Forest', 'Decision Tree', 'XGBoost'])

# ============================================================
# PREDICTION
# ============================================================
if st.button(" Predict Sentiment", type="primary", use_container_width=True):
    
    try:
        # Build feature dictionary in EXACT order as training
        input_data = {
            'order_status': order_status,
            'customer_zip_code_prefix': customer_zip_code_prefix,
            'customer_city': customer_city,
            'customer_state': customer_state,
            'order_item_id': order_item_id,
            'price': price,
            'freight_value': freight_value,
            'product_category_name': product_category_name,
            'product_name_lenght': product_name_lenght,
            'product_description_lenght': product_description_lenght,
            'product_photos_qty': product_photos_qty,
            'product_weight_g': product_weight_g,
            'product_length_cm': product_length_cm,
            'product_height_cm': product_height_cm,
            'product_width_cm': product_width_cm,
            'payment_sequential': payment_sequential,
            'payment_type': payment_type,
            'payment_installments': payment_installments,
            'payment_value': payment_value,
            'product_category_name_english': product_category_name_english,
            'delivery_days': delivery_days,
            'delivery_vs_estimated': delivery_vs_estimated,
            'approval_hours': approval_hours,
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Label encode categorical columns
        categorical_cols = ['order_status', 'customer_city', 'customer_state', 
                           'product_category_name', 'payment_type', 
                           'product_category_name_english']
        
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories gracefully
                if input_df[col].values[0] in le.classes_:
                    input_df[col] = le.transform(input_df[col])
                else:
                    # Use the most frequent class as fallback
                    input_df[col] = 0
                    st.warning(f" '{input_df[col].values[0]}' not seen during training for '{col}'. Using default value.")

        # Convert all to numeric
        input_df = input_df.astype(float)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make prediction based on model choice
        if model_choice == 'Random Forest':
            prediction = rf_model.predict(input_scaled)[0]
            model_used = "Random Forest"
        elif model_choice == 'Decision Tree':
            prediction = dt_model.predict(input_scaled)[0]
            model_used = "Decision Tree"
        elif model_choice == 'XGBoost':
            # XGBoost uses 0-indexed labels, shift back
            prediction = xgb_model.predict(input_scaled)[0] + 1
            model_used = "XGBoost"
        else:
            prediction = best_model.predict(input_scaled)[0]
            model_used = "Best Model"

        prediction = int(prediction)

        # Sentiment mapping
        sentiment_map = {
            1: ('😡 Very Negative', 'red'),
            2: ('😞 Negative', 'orange'),
            3: ('😐 Neutral', 'gray'),
            4: ('😊 Positive', 'blue'),
            5: ('🤩 Very Positive', 'green')
        }

        sentiment, color = sentiment_map.get(prediction, ('Unknown', 'gray'))

        # Display Results
        st.markdown("---")
        st.subheader(" Prediction Result")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(label="Predicted Score", value=f"{prediction} / 5")
        
        with res_col2:
            st.metric(label="Sentiment", value=sentiment)
        
        with res_col3:
            st.metric(label="Model Used", value=model_used)

        # Visual score bar
        st.markdown("### Score Scale")
        score_cols = st.columns(5)
        for i, col in enumerate(score_cols, 1):
            if i == prediction:
                col.markdown(f"<div style='text-align:center; padding:10px; background-color:#64ffda; border-radius:10px; color:black;'><b>{i} {'😡😞😐😊🤩'[i-1]}</b></div>", unsafe_allow_html=True)
            else:
                col.markdown(f"<div style='text-align:center; padding:10px; background-color:#333; border-radius:10px;'>{i} {'😡😞😐😊🤩'[i-1]}</div>", unsafe_allow_html=True)

        # Show all model predictions
        st.markdown("### All Model Predictions")
        all_preds = {
            'Random Forest': int(rf_model.predict(input_scaled)[0]),
            'Decision Tree': int(dt_model.predict(input_scaled)[0]),
            'XGBoost': int(xgb_model.predict(input_scaled)[0] + 1),
        }
        
        pred_df = pd.DataFrame({
            'Model': all_preds.keys(),
            'Predicted Score': all_preds.values(),
            'Sentiment': [sentiment_map[v][0] for v in all_preds.values()]
        })
        st.table(pred_df)

    except Exception as e:
        st.error(f" Prediction Error: {str(e)}")
        st.info(" Make sure all model files are in the 'models/' folder and features match training data.")


# --- Sidebar Info ---
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts customer review scores (1-5) 
    for the **Olist Brazilian E-Commerce** dataset.
    
    **Models Available:**
    -  Random Forest
    -  Decision Tree
    -  XGBoost
    
    **Pipeline:**
    1. Label Encoding (categorical)
    2. Standard Scaling (numerical)
    3. SMOTE (class balancing)
    4. Model Prediction
    """)
    
    st.markdown("---")
    st.header("Feature Importance")
    st.markdown("""
    Key features affecting sentiment:
    - Delivery time vs estimated
    - Product price
    - Payment value
    - Freight value
    - Product description length
    """)

