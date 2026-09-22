import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Page configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('model/customer_segments.csv')
        return df
    except FileNotFoundError:
        st.error("Data file 'model/customer_segments.csv' not found. Please run the training notebook first.")
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/kmeans_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'model/kmeans_model.pkl' not found. Please run the training notebook first.")
        return None

def main():
    st.title("📱 Smartphone Customer Segmentation Dashboard")
    st.markdown("""
    This dashboard visualizes the natural customer segments discovered using the **K-Means Clustering** algorithm.
    Segments are based on **Age** and **Annual Income**.
    """)

    df = load_data()
    model = load_model()

    if df is not None and model is not None:
        # --- Sidebar / Filters ---
        st.sidebar.header("Dashboard Settings")
        show_raw_data = st.sidebar.checkbox("Show Raw Data")

        # --- Metrics Row ---
        st.subheader("📊 Key Metrics")

        # Calculate Silhouette Score
        score = silhouette_score(df[['Age', 'Annual_Income_k']], df['Cluster'])

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        with col3:
            st.metric("Avg Annual Income", f"${df['Annual_Income_k'].mean():.1f}k")
        with col4:
            st.metric("Number of Clusters", len(df['Cluster'].unique()))
        with col5:
            st.metric("Silhouette Score", f"{score:.2f}")

        # --- Visualization Row ---
        st.subheader("🔍 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x='Age',
            y='Annual_Income_k',
            hue='Cluster',
            palette='viridis',
            s=100,
            ax=ax
        )
        ax.set_title("Customer Segments: Age vs Annual Income")
        st.pyplot(fig)

        # --- Analysis Row ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📈 Cluster Characteristics")
            cluster_summary = df.groupby('Cluster').agg({
                'Age': 'mean',
                'Annual_Income_k': 'mean',
                'Cluster': 'count'
            }).rename(columns={'Cluster': 'Count'}).reset_index()
            st.dataframe(cluster_summary.style.format({
                'Age': '{:.1f}',
                'Annual_Income_k': '{:.1f}',
                'Count': '{:d}'
            }), use_container_width=True)

        with col_right:
            st.subheader("💡 Marketing Insights")
            # Dynamic insights based on cluster means
            for idx, row in cluster_summary.iterrows():
                with st.expander(f"Segment {int(row['Cluster'])}"):
                    st.write(f"**Profile:** {row['Count']} customers")
                    st.write(f"- Avg Age: {row['Age']:.1f}")
                    st.write(f"- Avg Income: ${row['Annual_Income_k']:.1f}k")

                    if row['Annual_Income_k'] > df['Annual_Income_k'].mean() and row['Age'] < df['Age'].mean():
                        st.info("🚀 **Strategy:** High-earning young professionals. Target with trendy, premium tech gadgets.")
                    elif row['Annual_Income_k'] > df['Annual_Income_k'].mean() and row['Age'] >= df['Age'].mean():
                        st.info("💼 **Strategy:** Affluent mature customers. Target with reliability, premium features, and status symbols.")
                    elif row['Annual_Income_k'] < df['Annual_Income_k'].mean() and row['Age'] < df['Age'].mean():
                        st.info("🎓 **Strategy:** Budget-conscious students/young adults. Target with value-for-money and financing options.")
                    else:
                        st.info("🛒 **Strategy:** General mass market. Target with balanced features and competitive pricing.")

        # --- Raw Data Section ---
        if show_raw_data:
            st.subheader("📋 Raw Data")
            st.write(df)

    st.divider()
    st.caption("Developed for the Smartphone Marketing Team | Powered by K-Means")

if __name__ == "__main__":
    main()
