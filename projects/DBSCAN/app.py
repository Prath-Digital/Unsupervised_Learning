import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="DBSCAN Anomaly Detection Dashboard", layout="wide")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv('model/segments-anomalies.csv')

@st.cache_resource
def load_model():
    return joblib.load('model/model.pkl')

try:
    df = load_data()
    model_data = load_model()
    scaler = model_data['scaler']
    dbscan = model_data['dbscan']
    feature_names = model_data['feature_names']
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# --- Sidebar: Prediction ---
st.sidebar.header("Predict Segment for New Data")
with st.sidebar.form("prediction_form"):
    st.write("Enter feature values:")
    input_values = {}
    for feat in feature_names:
        input_values[feat] = st.number_input(f"{feat}", value=0.0, format="%.2f")

    submit_button = st.form_submit_button("Predict Segment")

if submit_button:
    # Prepare input for prediction
    input_df = pd.DataFrame([input_values])

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Since DBSCAN is transductive, we'll find the nearest neighbor in the training set
    # We need the scaled training data for this.
    # Let's recalculate or assume we can use the data in the dataframe.
    # For better performance, we should have saved the scaled data in the pickle.
    # But for now, let's use the training data.

    # Get original features (excluding user_id and target columns)
    train_features = df[feature_names]
    train_scaled = scaler.transform(train_features)

    # Find the index of the nearest neighbor
    distances = np.linalg.norm(train_scaled - input_scaled, axis=1)
    nearest_idx = np.argmin(distances)

    predicted_segment = df.iloc[nearest_idx]['segment']
    is_anomaly = df.iloc[nearest_idx]['is_anomaly']

    st.sidebar.markdown("---")
    if is_anomaly:
        st.sidebar.error(f"Predicted Segment: {predicted_segment} (⚠️ ANOMALY)")
    else:
        st.sidebar.success(f"Predicted Segment: {predicted_segment}")

# --- Main Dashboard ---
st.title("📊 Anomaly Detection Dashboard")
st.markdown("Insights from DBSCAN Clustering")

# --- Row 1: Metrics ---
col1, col2, col3, col4 = st.columns(4)
total_count = len(df)
anomaly_count = df['is_anomaly'].sum()
unique_segments = df['segment'].nunique()

col1.metric("Total Records", total_count)
col2.metric("Anomalies Found", anomaly_count, delta=f"{anomaly_count/total_count:.1%}", delta_color="inverse")
col3.metric("Unique Segments", unique_segments)
col4.metric("Anomaly %", f"{anomaly_count/total_count:.1%}")

st.markdown("---")

# --- Row 2: Visualizations ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Segment Distribution")
    fig, ax = plt.subplots()
    segment_counts = df['segment'].value_counts().sort_index()
    # Replace -1 with 'Anomaly' for better labeling
    labels = [str(s) if s != -1 else 'Anomaly' for s in segment_counts.index]
    ax.pie(segment_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(segment_counts)))
    st.pyplot(fig)

with col_right:
    st.subheader("Feature Relationship")
    # Select two features for scatter plot
    feat_x = st.selectbox("X Axis", feature_names, index=0)
    feat_y = st.selectbox("Y Axis", feature_names, index=1)

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=feat_x, y=feat_y, hue='segment', palette='viridis', ax=ax)
    # Highlight anomalies
    anomalies = df[df['is_anomaly']]
    ax.scatter(anomalies[feat_x], anomalies[feat_y], color='red', marker='x', s=100, label='Anomalies')
    st.pyplot(fig)

st.markdown("---")

# --- Row 3: Data Table ---
st.subheader("Processed Data")
st.dataframe(df, use_container_width=True)
