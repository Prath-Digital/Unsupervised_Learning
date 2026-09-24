import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load models and data
@st.cache_resource
def load_resources():
    try:
        # Load the unified model package
        model_package = joblib.load('model/model.pkl')
        scaler = model_package['scaler']
        knn = model_package['knn']
        features = model_package['features']

        # Load the segments data
        df = pd.read_csv('model/segments.csv')

        return scaler, knn, df, features
    except Exception as e:
        return None, None, None, None

scaler, knn, df, features = load_resources()

if scaler is None:
    st.error("❌ Error: Could not load models or data. Please ensure 'train.ipynb' has been run successfully to generate 'model/model.pkl' and 'model/segments.csv'.")
    st.stop()

# --- Sidebar: Custom Input ---
st.sidebar.header("🔍 Test Custom Input")
st.sidebar.write("Enter customer details to predict their segment.")

# Use the feature names from the model package
col1_label = features[0].replace('_', ' ')
col2_label = features[1].replace('_', ' ')

input_val1 = st.sidebar.number_input(col1_label, min_value=0.0, max_value=200.0, value=50.0, step=1.0)
input_val2 = st.sidebar.number_input(col2_label, min_value=0.0, max_value=100.0, value=50.0, step=1.0)

if st.sidebar.button("Predict Segment"):
    # Prepare input for prediction
    user_data = np.array([[input_val1, input_val2]])
    user_data_scaled = scaler.transform(user_data)
    prediction = knn.predict(user_data_scaled)[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Result")
    st.sidebar.success(f"Predicted Segment: **{int(prediction)}**")
    st.sidebar.info(f"Based on inputs: {col1_label} {input_val1}, {col2_label} {input_val2}")

# --- Main Area ---
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("""
This dashboard visualizes customer segments discovered using **Hierarchical Clustering**.
You can explore the existing segments or use the sidebar to predict a segment for a new customer.
""")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📈 Segment Metrics", "📋 Raw Data"])

with tab1:
    st.subheader("Customer Segments Scatter Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='viridis', s=100, ax=ax)
    ax.set_title("Income vs Spending Score by Segment")
    st.pyplot(fig)

    st.subheader("Cluster Density Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x=features[0], y=features[1], hue='Cluster', fill=True, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_title("Probability Density of Customer Segments")
    st.pyplot(fig2)

with tab2:
    st.subheader("Segment Statistical Summary")
    # Group by cluster and get mean
    stats = df.groupby('Cluster').agg({
        features[0]: ['mean', 'std'],
        features[1]: ['mean', 'std'],
        'Cluster': 'count'
    }).reset_index()

    # Flatten columns
    stats.columns = ['Cluster', f'{features[0]} Mean', f'{features[0]} Std', f'{features[1]} Mean', f'{features[1]} Std', 'Count']

    # Display stats table
    st.dataframe(stats.style.format({
        f'{features[0]} Mean': '${:.2f}k',
        f'{features[0]} Std': '${:.2f}k',
        f'{features[1]} Mean': '{:.2f}',
        f'{features[1]} Std': '{:.2f}',
        'Count': '{:,.0f}'
    }).highlight_max(axis=0, subset=['Count']))

    # Detailed Expander for each segment
    st.markdown("---")
    st.subheader("Detailed Segment Profiles")
    cols = st.columns(len(stats['Cluster'].unique()))

    for i, (_, row) in enumerate(stats.iterrows()):
        with cols[i]:
            st.metric(label=f"Segment {int(row['Cluster'])}", value=f"{int(row['Count'])} customers")
            st.write(f"**Avg {features[0].replace('_', ' ')}:** ${row[f'{features[0]} Mean']:.1f}k")
            st.write(f"**Avg {features[1].replace('_', ' ')}:** {row[f'{features[1]} Mean']:.1f}")

with tab3:
    st.subheader("Dataset Overview")
    st.dataframe(df)

# Footer
st.markdown("---")
st.caption("Powered by Hierarchical Clustering & Streamlit")
