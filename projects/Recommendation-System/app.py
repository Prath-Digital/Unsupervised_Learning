import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="Recommendation System", layout="wide")

st.title("🚀 Recommendation Systems Showcase")

# Load all data and models
@st.cache_resource
def load_apriori():
    with open('Apriory/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Apriory/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    df = pd.read_csv('data/apriori_market_basket_10000.csv')
    return model, metrics, df

@st.cache_resource
def load_content_based():
    with open('Content-Based/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Content-Based/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    df = pd.read_csv('data/content_based_products_10000.csv')
    return model, metrics, df

@st.cache_resource
def load_collaborative():
    with open('Collaborative/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Collaborative/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    df = pd.read_csv('data/collaborative_user_item_10000.csv')
    return model, metrics, df

@st.cache_resource
def load_hybrid():
    with open('Hybrid/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('Hybrid/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    df = pd.read_csv('data/hybrid_recommendation_10000.csv')
    return model, metrics, df

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Apriori", "Content-Based", "Collaborative", "Hybrid"])

# --- APRIORI TAB ---
with tab1:
    st.header("🛒 Apriori (Association Rules)")
    try:
        model, metrics, df = load_apriori()

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Support", f"{metrics['avg_support']:.4f}")
        col2.metric("Avg Confidence", f"{metrics['avg_confidence']:.4f}")
        col3.metric("Avg Lift", f"{metrics['avg_lift']:.4f}")

        st.subheader("Raw Transaction Data")
        st.dataframe(df.head(100))

        st.subheader("Test Recommendation")
        product_list = sorted(df['item'].unique())
        selected_product = st.selectbox("Select a product you bought:", product_list)

        if selected_product:
            rules = model['rules']
            # Find rules where selected_product is in antecedents
            # Note: antecedents are frozensets
            recommendations = rules[rules['antecedents'].apply(lambda x: selected_product in x)]

            if not recommendations.empty:
                rec_products = recommendations['consequents'].apply(lambda x: list(x)[0]).unique()
                st.success(f"Because you bought **{selected_product}**, we recommend:")
                for p in rec_products[:5]:
                    st.write(f"- {p}")
            else:
                st.warning("No direct associations found for this product.")
    except Exception as e:
        st.error(f"Error loading Apriori: {e}")

# --- CONTENT-BASED TAB ---
with tab2:
    st.header("🔍 Content-Based Filtering")
    try:
        model, metrics, df = load_content_based()

        col1, col2 = st.columns(2)
        col1.metric("Avg Similarity", f"{metrics['avg_similarity']:.4f}")
        col2.metric("Total Products", metrics['num_products'])

        st.subheader("Raw Product Data")
        st.dataframe(df.head(100))

        st.subheader("Test Recommendation")
        product_list = sorted(df['item_name'].unique())
        selected_product = st.selectbox("Select a product you are interested in:", product_list)

        if selected_product:
            idx = model['item_indices'][selected_product]
            sim_scores = list(enumerate(model['cosine_sim'][idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6] # Top 5 similar (excluding itself)

            item_indices = [i[0] for i in sim_scores]
            st.success(f"Since you liked **{selected_product}**, you might also like:")
            for i in item_indices:
                st.write(f"- {df.iloc[i]['item_name']}")
    except Exception as e:
        st.error(f"Error loading Content-Based: {e}")

# --- COLLABORATIVE TAB ---
with tab3:
    st.header("🤝 Collaborative Filtering")
    try:
        model, metrics, df = load_collaborative()

        col1, col2, col3 = st.columns(3)
        col1.metric("Users", int(metrics['num_users']))
        col2.metric("Items", int(metrics['num_items']))
        col3.metric("Sparsity", f"{metrics['sparsity']:.4f}")

        st.subheader("Raw Interaction Data")
        st.dataframe(df.head(100))

        st.subheader("Test Recommendation")
        user_ids = sorted(df['user_id'].unique())
        selected_user = st.selectbox("Select a User ID:", user_ids)

        if selected_user:
            # Find similar items using NearestNeighbors
            # The model was trained on item_id columns
            user_idx = model['user_item_matrix'].index.get_loc(selected_user)
            user_vector = model['user_item_matrix'].iloc[user_idx].values.reshape(1, -1)

            # Find items the user has interacted with
            user_interactions = df[df['user_id'] == selected_user]
            if not user_interactions.empty:
                last_item_id = user_interactions.iloc[-1]['item_id']
                item_idx = model['item_ids'].index(last_item_id)

                distances, indices = model['model'].kneighbors(
                    model['user_item_matrix'].values.T[item_idx].reshape(1, -1),
                    n_neighbors=6
                )

                rec_indices = indices[0][1:] # exclude itself
                rec_item_ids = [model['item_ids'][i] for i in rec_indices]

                st.success(f"Based on your recent interest in item {last_item_id}, we recommend:")
                for rid in rec_item_ids:
                    st.write(f"- Item ID: {rid}")
            else:
                st.warning("User has no interaction history.")
    except Exception as e:
        st.error(f"Error loading Collaborative: {e}")

# --- HYBRID TAB ---
with tab4:
    st.header("🧬 Hybrid Recommendation")
    try:
        model, metrics, df = load_hybrid()

        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", metrics['num_samples'])
        col2.metric("Avg Rating", f"{metrics['avg_rating']:.2f}")
        col3.metric("Purchase Rate", f"{metrics['purchase_rate']:.2%}")

        st.subheader("Raw Hybrid Data")
        st.dataframe(df.head(100))

        st.subheader("Test Recommendation (Predict Purchase Probability)")
        with st.form("hybrid_form"):
            user_age = st.number_input("User Age", min_value=18, max_value=100, value=30)
            annual_income = st.number_input("Annual Income", min_value=0, max_value=500000, value=50000)
            item_category = st.selectbox("Item Category", df['item_category'].unique())
            item_price = st.number_input("Item Price", min_value=0.0, max_value=10000.0, value=100.0)
            item_rating = st.slider("Item Rating", 1, 5, 3)
            views = st.number_input("Views", min_value=0, max_value=1000, value=10)
            cart_additions = st.number_input("Cart Additions", min_value=0, max_value=100, value=1)

            submit = st.form_submit_button("Predict Purchase Probability")

        if submit:
            # Prepare input for model
            # Need to match training features
            input_df = pd.DataFrame([{
                'user_age': user_age,
                'annual_income': annual_income,
                'item_price': item_price,
                'item_rating': item_rating,
                'views': views,
                'cart_additions': cart_additions
            }])

            # One-hot encode category
            category_cols = pd.get_dummies(df['item_category']).columns
            for cat in category_cols:
                input_df[cat] = 1 if cat == item_category else 0

            # Ensure all columns are present and in correct order
            try:
                # The model was trained on df_encoded.drop(['user_id', 'item_id', 'purchased'], axis=1)
                # We must match that exactly.
                expected_features = model['features']
                for col in expected_features:
                    if col not in input_df.columns:
                        input_df[col] = 0

                input_df = input_df[expected_features]

                prob = model['model'].predict(input_df)[0]
                st.info(f"Predicted Purchase Probability: **{prob:.2%}**")
                if prob > 0.5:
                    st.success("High probability of purchase! Recommend this item.")
                else:
                    st.warning("Low probability of purchase.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    except Exception as e:
        st.error(f"Error loading Hybrid: {e}")
