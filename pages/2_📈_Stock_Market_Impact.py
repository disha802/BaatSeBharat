import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Stock Market Impact",
    page_icon="📈",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#1f77b4;'>📈 Topic–Stock Impact Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Analyze how Mann Ki Baat topics correlate with Indian stock market sectors")
st.markdown("---")

# Sidebar
st.sidebar.header("🗂 Data Configuration")

topic_file = st.sidebar.text_input(
    "Quarterly Topics CSV path",
    value="data/quarterly_topics.csv",
    help="Path to the file exported from Topic Modeling"
)

# Load topic data
try:
    topics_df = pd.read_csv(topic_file)
    st.success(f"✅ Loaded {len(topics_df)} quarterly topic entries.")
except Exception as e:
    st.error(f"Could not load topic file: {e}")
    st.info("💡 Please run Topic Modeling first and export the data to data/quarterly_topics.csv")
    st.stop()

# Convert Quarter string to datetime
try:
    topics_df["Quarter_Date"] = pd.PeriodIndex(topics_df["Quarter"], freq="Q").to_timestamp()
    st.info(f"📅 Topic data spans: {topics_df['Quarter_Date'].min().strftime('%Y-%m-%d')} to {topics_df['Quarter_Date'].max().strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"Error parsing Quarter column: {e}")
    st.error("Expected format: '2020Q1', '2020Q2', etc.")
    st.stop()

# Topic-sector mapping
st.sidebar.markdown("---")
st.sidebar.header("🏢 Topic–Sector Mapping")

topic_to_companies = {
    "Digital India & E-Governance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "Innovation & Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    "Finance & Economy": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
    "Energy & Power": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS"],
    "Manufacturing & Make in India": ["TATASTEEL.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "M&M.NS"],
    "Agriculture & Rural Economy": ["ITC.NS", "UPL.NS", "COROMANDEL.NS"],
    "Infrastructure & Development": ["LT.NS", "ULTRACEMCO.NS"],
    "Healthcare & Pandemic Response": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    "Social Empowerment & Youth": ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS"],
}

tickers = sorted(set(sum(topic_to_companies.values(), [])))
st.sidebar.write(f"Tracking **{len(tickers)} companies** across **{len(topic_to_companies)} topics**")

# Fetch stock data
st.info("📊 Fetching stock data from Yahoo Finance... (This may take 30–60s)")

start_date = topics_df["Quarter_Date"].min()
end_date = topics_df["Quarter_Date"].max() + pd.DateOffset(months=3)

try:
    data_download = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        interval="3mo",
        auto_adjust=False,
        progress=False
    )
    
    # Extract Adj Close or Close gracefully
    if "Adj Close" in data_download.columns.get_level_values(0):
        stock_df = data_download["Adj Close"]
    else:
        stock_df = data_download["Close"]
    
    if isinstance(stock_df, pd.Series):
        stock_df = stock_df.to_frame(name=tickers[0])
    
    stock_df = stock_df.dropna(how="all")
    
    missing_tickers = [t for t in tickers if t not in stock_df.columns]
    if missing_tickers:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(missing_tickers)}")
    
    valid_tickers = [t for t in tickers if t in stock_df.columns]
    st.success(f"✅ Retrieved {stock_df.shape[0]} quarterly records for {len(valid_tickers)}/{len(tickers)} companies.")
    st.info(f"📅 Stock data spans: {stock_df.index.min().strftime('%Y-%m-%d')} to {stock_df.index.max().strftime('%Y-%m-%d')}")
    
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Compute returns
returns_df = stock_df.pct_change().dropna(how="all")
returns_df.index = pd.to_datetime(returns_df.index)
returns_df["Quarter_Period"] = returns_df.index.to_period("Q")

# Aggregate sector returns
sector_returns = {}
for topic, comps in topic_to_companies.items():
    valid_comps = [c for c in comps if c in returns_df.columns]
    if valid_comps:
        sector_returns[topic] = returns_df[valid_comps].mean(axis=1)

sector_df = pd.DataFrame(sector_returns).dropna(how="all")
sector_df["Quarter_Period"] = returns_df["Quarter_Period"]

# Merge topics + stock returns
st.markdown("## 🔗 Merging Topic and Stock Data...")

topics_df["Quarter_Period"] = topics_df["Quarter_Date"].dt.to_period("Q")

merged_results = []

for topic, comps in topic_to_companies.items():
    if topic not in sector_df.columns:
        continue
    
    topic_strength = topics_df.loc[topics_df["Topic_Label"] == topic].copy()
    if topic_strength.empty:
        continue

    for _, row in topic_strength.iterrows():
        q_period = row["Quarter_Period"]
        
        matching_stock = sector_df[sector_df["Quarter_Period"] == q_period]
        
        if not matching_stock.empty:
            topic_rank = 6 - row["Rank"]
            sector_return = matching_stock[topic].iloc[0]
            
            merged_results.append({
                "Quarter": row["Quarter"],
                "Quarter_Date": row["Quarter_Date"],
                "Topic": topic,
                "Topic_Strength": topic_rank,
                "Sector_Return": sector_return
            })

merged_df = pd.DataFrame(merged_results).dropna()

st.success(f"✅ Successfully merged {len(merged_df)} topic-stock data points across {merged_df['Quarter'].nunique()} quarters")

# Correlation analysis
st.markdown("## 🔍 Topic–Sector Correlation Analysis")

if merged_df.empty:
    st.error("❌ No overlapping quarters found between topic data and stock data.")
    st.write("**Troubleshooting Tips:**")
    st.write("1. Check that your quarterly_topics.csv has a 'Quarter' column with format like '2020Q1'")
    st.write("2. Verify the date ranges overlap between topics and stock data")
    st.write("3. Ensure topic labels match exactly with the predefined categories")
    st.stop()

corr_summary = (
    merged_df.groupby("Topic")[["Topic_Strength", "Sector_Return"]]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
    .rename(columns={"Sector_Return": "Correlation"})
    .drop(columns=["level_1"])
)

st.dataframe(corr_summary.style.format({"Correlation": "{:.2f}"}), use_container_width=True)

fig = px.bar(
    corr_summary,
    x="Topic",
    y="Correlation",
    title="Correlation between Topic Strength and Sector Returns",
    color="Correlation",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
)
fig.update_layout(height=500, xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Temporal Trend Comparison
st.markdown("## ⏳ Topic Strength vs Sector Performance Over Time")

selected_topic = st.selectbox(
    "Select a Topic to View Trends:",
    options=sorted(topic_to_companies.keys())
)

if selected_topic in merged_df["Topic"].unique():
    subset = merged_df[merged_df["Topic"] == selected_topic].sort_values("Quarter_Date")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=subset["Quarter_Date"], y=subset["Topic_Strength"],
        name="Topic Strength", mode="lines+markers", yaxis="y1",
        line=dict(color='#1f77b4', width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=subset["Quarter_Date"], y=subset["Sector_Return"],
        name="Sector Return", mode="lines+markers", yaxis="y2",
        line=dict(color='#ff7f0e', width=3)
    ))

    fig2.update_layout(
        title=f"📅 {selected_topic}: Topic Strength vs Sector Return Over Time",
        xaxis_title="Quarter",
        yaxis=dict(title="Topic Strength", side="left", showgrid=False),
        yaxis2=dict(title="Sector Return (%)", side="right", overlaying="y", showgrid=False),
        height=500,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📊 View Data Table"):
        display_cols = ["Quarter", "Topic_Strength", "Sector_Return"]
        st.dataframe(subset[display_cols].style.format({"Topic_Strength": "{:.2f}", "Sector_Return": "{:.2%}"}))
else:
    st.warning("Selected topic not found in merged dataset.")

# Export results
st.markdown("## 💾 Export Analysis Results")
csv_data = merged_df.to_csv(index=False)
st.download_button(
    "Download Merged Topic–Stock Data (CSV)",
    data=csv_data,
    file_name="topic_stock_correlation.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built with Streamlit | Topic–Stock Impact Analysis</div>",
    unsafe_allow_html=True
)