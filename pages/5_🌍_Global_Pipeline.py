import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.models import DualPredictiveModels

st.set_page_config(
    page_title="Global Intelligence Pipeline",
    page_icon="🌍",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#27ae60;'>🌍 Global Intelligence Pipeline</h1>", unsafe_allow_html=True)
st.markdown("### End-to-End Analysis: Speech Acquisition to Hybrid Market Prediction")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("🛠️ Pipeline Configuration")

leader = st.sidebar.selectbox(
    "Select Global Leader",
    list(DataFetcher.LEADER_MAP.keys())
)

index = st.sidebar.selectbox(
    "Associated Market Index",
    list(DataFetcher.INDEX_MAP.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
window_size = st.sidebar.slider("Lookback Window (Days)", 10, 60, 20)
horizon = st.sidebar.slider("Prediction Horizon (Days Ahead)", 1, 30, 5)

# Step 1: Data Acquisition
st.header("1️⃣ Data Acquisition")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Market Data")
    if st.button("📥 Fetch Index Data"):
        with st.spinner(f"Fetching {index}..."):
            end = datetime.now()
            start = end - timedelta(days=365*5) # 5 years
            df_market = DataFetcher.fetch_stock_data(index, start, end)
            if not df_market.empty:
                st.session_state.market_data = df_market
                st.success(f"Retrieved {len(df_market)} trading days for {index}")
                st.line_chart(df_market['Price'])
            else:
                st.error("Failed to fetch market data.")

with col2:
    st.subheader("Speech Data")
    speeches = DataFetcher.list_available_speeches(leader)
    st.write(f"**Leader:** {leader}")
    st.write(f"**Speeches Found:** {len(speeches)}")
    
    if leader == "Nirmala Sitharaman (India)":
        st.info("💡 Note: Budget speeches are usually available in Feb each year.")
    
    # Check for existing processed topic data
    topic_file = "data/episode_topics.csv"
    if os.path.exists(topic_file):
        st.session_state.topic_data = pd.read_csv(topic_file)
        st.session_state.topic_data['date'] = pd.to_datetime(st.session_state.topic_data['date'])
        st.success("✅ Processed topic modeling data found in `data/`")
    else:
        st.warning("⚠️ No processed topic data found. Please run Topic Modeling first.")

# Step 2: Dual Model Training
st.header("2️⃣ Dual-Model Prediction")

if 'market_data' in st.session_state:
    if st.button("🚀 Run Pipeline & Compare Models"):
        m_df = st.session_state.market_data
        t_df = st.session_state.topic_data if 'topic_data' in st.session_state else None
        
        models = DualPredictiveModels(window_size=window_size, prediction_horizon=horizon)
        
        # Prepare Features
        df_a, df_b = models.prepare_features(m_df, t_df)
        
        # Model A Training
        f_cols_a = [f'price_lag_{i}' for i in range(1, window_size + 1)]
        pred_a, metrics_a = models.train_and_predict(df_a, f_cols_a)
        
        # Model B Training
        if df_b is not None:
            topic_cols = [c for c in t_df.columns if c not in ['episode', 'date', 'quarter']]
            f_cols_b = f_cols_a + topic_cols
            pred_b, metrics_b = models.train_and_predict(df_b, f_cols_b)
        else:
            pred_b, metrics_b = None, None

        # Display Comparison
        c1, c2 = st.columns(2)
        
        with c1:
            st.metric("Model A (Stock Only) Prediction", f"{pred_a:.2f}")
            if metrics_a:
                st.write(f"RMSE: {metrics_a['RMSE']:.4f}")
                st.write(f"R² Score: {metrics_a['R2']:.4f}")

        with c2:
            if pred_b:
                st.metric("Model B (Rhetoric-Enhanced) Prediction", f"{pred_b:.2f}")
                st.write(f"RMSE: {metrics_b['RMSE']:.4f}")
                st.write(f"Average Delta: {pred_b - pred_a:.2f}")
            else:
                st.warning("Model B unavailable (No topic data)")

        # Final Projection Chart
        st.subheader("📊 Comparative Projections")
        
        last_date = m_df.index[-1]
        future_date = last_date + timedelta(days=horizon)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m_df.index[-50:], y=m_df['Price'][-50:], name="Historical", line=dict(color='black')))
        fig.add_trace(go.Scatter(x=[last_date, future_date], y=[m_df['Price'].iloc[-1], pred_a], name="Model A", line=dict(dash='dash', color='blue')))
        if pred_b:
             fig.add_trace(go.Scatter(x=[last_date, future_date], y=[m_df['Price'].iloc[-1], pred_b], name="Model B (Hybrid)", line=dict(dash='dash', color='green')))
        
        fig.update_layout(title=f"Price Prediction for {future_date.date()}", height=500)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please fetch market data in Step 1 to enable training.")

import os
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Global Intelligence Pipeline | End-to-End Rhetoric & Market Analysis</div>", unsafe_allow_html=True)
