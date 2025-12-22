import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="Stock Market Impact",
    page_icon="📈",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#1f77b4;'>📈 Topic–Stock Impact Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Analyze how Mann Ki Baat topics correlate with Indian stock market sectors")
st.markdown("---")

# Check if model has been trained
if 'model_trained' not in st.session_state or not st.session_state.model_trained:
    st.error("⚠️ No topic model data found!")
    st.info("👉 Please go to the **Topic Modeling** page first and train the model.")
    st.stop()

# Get data from session state
df = st.session_state.df
topics = st.session_state.topics
topic_labels = st.session_state.topic_labels
topic_cols = st.session_state.topic_cols

# Generate quarterly topics data from session state
st.sidebar.header("🗂 Data Configuration")
st.success(f"✅ Using data from trained model ({len(df)} episodes)")

# Create quarterly topics dataframe
quarterly_results = []
for quarter, group in df.groupby("quarter"):
    if pd.isna(quarter):
        continue
    mean_dist = group[topic_cols].mean().sort_values(ascending=False)
    top_topic_ids = mean_dist.index[:5]
    
    for rank, tid in enumerate(top_topic_ids, start=1):
        idx = int(tid.split("_")[1])
        label = topic_labels.get(idx, f"Topic {idx+1}")
        top_words = ", ".join(topics[idx][:8])
        quarterly_results.append({
            "Quarter": str(quarter),
            "Rank": rank,
            "Topic_Label": label,
            "Top_Words": top_words
        })

topics_df = pd.DataFrame(quarterly_results)

# Convert Quarter string to datetime
try:
    topics_df["Quarter_Date"] = pd.PeriodIndex(topics_df["Quarter"], freq="Q").to_timestamp()
    st.info(f"📅 Topic data spans: {topics_df['Quarter_Date'].min().strftime('%Y-%m-%d')} to {topics_df['Quarter_Date'].max().strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"Error parsing Quarter column: {e}")
    st.stop()

# Topic-sector mapping
st.sidebar.markdown("---")
st.sidebar.header("🏢 Topic–Sector Mapping")

topic_to_companies = {
    "Digital India & E-Governance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "Innovation & Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    "Finance & Economy": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
    "Energy & Power": ["RELIANCE.NS", "NTPC.NS", "ADANIPOWER.NS"],
    "Manufacturing & Make in India": ["TATASTEEL.NS", "MARUTI.NS", "BAJAJ-AUTO.BO", "M&M.NS"],
    "Agriculture & Rural Economy": ["ITC.NS", "UPL.NS", "COROMANDEL.NS"],
    "Infrastructure & Development": ["LT.NS", "ULTRACEMCO.NS"],
    "Healthcare & Pandemic Response": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    "Social Empowerment & Youth": ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.BO"],
    "Education & Learning": ["ITC.NS", "HINDUNILVR.NS"],
    "Yoga & Wellness": ["APOLLOHOSP.NS", "SUNPHARMA.NS"],
    "Environment & Water Conservation": ["ITC.NS", "TATASTEEL.NS"],
    "Culture & Rural Development": ["ITC.NS", "TITAN.NS"],
    "General / Mixed Theme": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
}

tickers = sorted(set(sum(topic_to_companies.values(), [])))
st.sidebar.write(f"Tracking **{len(tickers)} companies** across **{len(topic_to_companies)} topics**")

# Enhanced stock data fetching with retry logic
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_robust(tickers, start, end, max_retries=3):
    """Fetch stock data with enhanced error handling and retry logic"""
    
    # Ensure dates are strings in YYYY-MM-DD format
    if isinstance(start, pd.Timestamp):
        start = start.strftime('%Y-%m-%d')
    if isinstance(end, pd.Timestamp):
        end = end.strftime('%Y-%m-%d')
    
    all_data = {}
    failed = []
    
    # Split tickers into smaller batches to avoid rate limiting
    batch_size = 5
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_idx, batch in enumerate(ticker_batches):
        status_text.text(f"Fetching batch {batch_idx+1}/{len(ticker_batches)}: {', '.join(batch)}")
        
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Add delay between requests to avoid rate limiting
                if batch_idx > 0:
                    time.sleep(2)
                
                # Try batch download
                batch_df = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by='ticker',
                    threads=False  # Disable threading to be more conservative
                )
                
                if not batch_df.empty:
                    for ticker in batch:
                        try:
                            if len(batch) > 1 and ticker in batch_df.columns.get_level_values(0):
                                ticker_data = batch_df[ticker]['Close']
                            elif len(batch) == 1:
                                ticker_data = batch_df['Close']
                            else:
                                continue
                            
                            if not ticker_data.empty and not ticker_data.isna().all():
                                all_data[ticker] = ticker_data
                        except Exception as e:
                            st.warning(f"Error processing {ticker}: {str(e)}")
                    
                    success = True
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    status_text.text(f"Retry {retry_count}/{max_retries} for batch {batch_idx+1}...")
                    time.sleep(5 * retry_count)  # Exponential backoff
                else:
                    st.warning(f"Failed to fetch batch {batch_idx+1} after {max_retries} retries")
        
        # Update progress
        progress_bar.progress((batch_idx + 1) / len(ticker_batches))
    
    progress_bar.empty()
    status_text.empty()
    
    # Fallback: Try individual downloads for missing tickers
    missing = [t for t in tickers if t not in all_data]
    
    if missing and len(missing) < len(tickers) * 0.5:  # Only if less than 50% failed
        st.info(f"Attempting individual downloads for {len(missing)} missing tickers...")
        
        for idx, ticker in enumerate(missing):
            try:
                time.sleep(1)  # Rate limiting
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(start=start, end=end, auto_adjust=True)
                
                if not hist.empty and 'Close' in hist.columns:
                    all_data[ticker] = hist['Close']
                else:
                    failed.append(ticker)
            except Exception as e:
                failed.append(ticker)
            
            if (idx + 1) % 5 == 0:
                st.text(f"Progress: {idx+1}/{len(missing)}")
    else:
        failed.extend(missing)
    
    if not all_data:
        return None, failed
    
    # Combine all data
    combined_df = pd.DataFrame(all_data)
    
    # Resample to quarterly
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df = combined_df.resample('Q').last()
    combined_df = combined_df.dropna(how='all')
    
    return combined_df, failed

# Check for demo mode toggle
if 'use_demo_mode' not in st.session_state:
    st.session_state.use_demo_mode = False

# Demo mode section
if st.session_state.use_demo_mode:
    st.warning("🎮 **Demo Mode Active** - Using simulated stock data")
    
    # Generate simulated stock returns
    np.random.seed(42)
    quarters = pd.period_range(start=topics_df["Quarter_Date"].min(), 
                               end=topics_df["Quarter_Date"].max(), 
                               freq='Q')
    
    stock_df = pd.DataFrame(
        index=quarters.to_timestamp(),
        columns=tickers[:15]  # Use subset of tickers
    )
    
    # Simulate realistic returns with trends
    for col in stock_df.columns:
        trend = np.random.choice([0.02, 0.01, -0.01], 1)[0]
        stock_df[col] = 100 * (1 + trend + np.random.randn(len(stock_df)) * 0.08).cumprod()
    
    valid_tickers = list(stock_df.columns)
    st.success(f"✅ Generated {stock_df.shape[0]} quarterly records for {len(valid_tickers)} companies (DEMO)")
    st.info(f"📅 Stock data spans: {stock_df.index.min().strftime('%Y-%m-%d')} to {stock_df.index.max().strftime('%Y-%m-%d')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Real Data Again"):
            st.session_state.use_demo_mode = False
            st.rerun()
    with col2:
        if st.button("🎲 Regenerate Demo Data"):
            st.cache_data.clear()
            st.rerun()

else:
    # Real data fetching
    st.info("📊 Fetching stock data from Yahoo Finance...")
    st.write("⏱️ This may take 1-2 minutes. Please be patient...")

    # Prepare date range
    start_date = pd.Timestamp(topics_df["Quarter_Date"].min())
    end_date = pd.Timestamp(topics_df["Quarter_Date"].max()) + pd.DateOffset(months=3)
    
    try:
        stock_df, failed_tickers = fetch_stock_data_robust(tickers, start_date, end_date)
        
        if stock_df is None or stock_df.empty:
            st.error("❌ Could not retrieve stock data")
            
            with st.expander("📋 Possible Causes & Solutions"):
                st.write("""
                **Common Issues:**
                - Yahoo Finance API rate limiting (most common on Streamlit Cloud)
                - Network connectivity issues
                - Date range too old (pre-2019 data may be limited)
                - All tickers invalid or delisted
                
                **What to do:**
                1. Wait 2-3 minutes and refresh the page
                2. Try during off-peak hours (early morning IST)
                3. Use Demo Mode to explore functionality
                4. Check if tickers are valid on Yahoo Finance
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Retry Now"):
                    st.cache_data.clear()
                    st.rerun()
            with col2:
                if st.button("🎮 Use Demo Mode", type="primary"):
                    st.session_state.use_demo_mode = True
                    st.rerun()
            
            st.stop()
        
        if failed_tickers:
            st.warning(f"⚠️ Could not fetch data for {len(failed_tickers)}/{len(tickers)} tickers")
            with st.expander("View failed tickers"):
                st.write(failed_tickers)
                st.caption("Note: Some tickers may be invalid, delisted, or temporarily unavailable")
        
        valid_tickers = list(stock_df.columns)
        
        st.success(f"✅ Retrieved {stock_df.shape[0]} quarterly records for {len(valid_tickers)}/{len(tickers)} companies")
        st.info(f"📅 Stock data spans: {stock_df.index.min().strftime('%Y-%m-%d')} to {stock_df.index.max().strftime('%Y-%m-%d')}")
        
        # Option to switch to demo mode even after successful fetch
        with st.expander("🎮 Want to try Demo Mode instead?"):
            if st.button("Switch to Demo Mode"):
                st.session_state.use_demo_mode = True
                st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {str(e)}")
        
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("🎮 Use Demo Mode", type="primary"):
                st.session_state.use_demo_mode = True
                st.rerun()
        
        st.stop()

# Rest of the analysis code continues here...
# [Keep all the correlation analysis, visualization, and export code the same]

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
    st.write("1. Check date ranges overlap between topics and stock data")
    st.write("2. Ensure topic labels match the predefined categories")
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

col1, col2 = st.columns(2)

with col1:
    csv_data = merged_df.to_csv(index=False)
    st.download_button(
        "Download Merged Topic–Stock Data (CSV)",
        data=csv_data,
        file_name="topic_stock_correlation.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    quarterly_csv = topics_df.to_csv(index=False)
    st.download_button(
        "Download Quarterly Topics Data (CSV)",
        data=quarterly_csv,
        file_name="quarterly_topics_from_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built with Streamlit | Topic–Stock Impact Analysis</div>",
    unsafe_allow_html=True
)