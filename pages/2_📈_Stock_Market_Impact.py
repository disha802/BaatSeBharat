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

# Generate quarterly topics data
st.sidebar.header("🗂 Data Configuration")
st.success(f"✅ Using data from trained model ({len(df)} episodes)")

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

# Convert Quarter to datetime
try:
    topics_df["Quarter_Date"] = pd.PeriodIndex(topics_df["Quarter"], freq="Q").to_timestamp()
    st.info(f"📅 Topic data spans: {topics_df['Quarter_Date'].min().strftime('%Y-%m-%d')} to {topics_df['Quarter_Date'].max().strftime('%Y-%m-%d')}")
except Exception as e:
    st.error(f"Error parsing Quarter column: {e}")
    st.stop()

# UPDATED Topic-sector mapping with reliable tickers only
st.sidebar.markdown("---")
st.sidebar.header("🏢 Topic–Sector Mapping")

topic_to_companies = {
    "Digital India & E-Governance": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "Innovation & Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    "Finance & Economy": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
    "Energy & Power": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS"],
    "Manufacturing & Make in India": ["TATASTEEL.NS", "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"],
    "Agriculture & Rural Economy": ["ITC.NS", "UPL.NS", "PIDILITIND.NS"],
    "Infrastructure & Development": ["LT.NS", "ULTRACEMCO.NS", "GRASIM.NS"],
    "Healthcare & Pandemic Response": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    "Social Empowerment & Youth": ["HINDUNILVR.NS", "ITC.NS", "DABUR.NS"],
    "Education & Learning": ["ITC.NS", "HINDUNILVR.NS", "TITAN.NS"],
    "Yoga & Wellness": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
    "Environment & Water Conservation": ["ITC.NS", "TATASTEEL.NS", "GRASIM.NS"],
    "Culture & Rural Development": ["ITC.NS", "TITAN.NS", "HINDUNILVR.NS"],
    "General / Mixed Theme": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
}

tickers = sorted(set(sum(topic_to_companies.values(), [])))
st.sidebar.write(f"Tracking **{len(tickers)} companies** across **{len(topic_to_companies)} topics**")

# Enhanced stock fetching function
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_robust(tickers, start, end, max_retries=2):
    """Fetch stock data with robust error handling"""
    
    # Convert dates to strings
    if isinstance(start, pd.Timestamp):
        start = start.strftime('%Y-%m-%d')
    if isinstance(end, pd.Timestamp):
        end = end.strftime('%Y-%m-%d')
    
    all_data = {}
    failed = []
    
    # Use smaller batches
    batch_size = 3
    ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_idx, batch in enumerate(ticker_batches):
        status_text.text(f"📥 Fetching batch {batch_idx+1}/{len(ticker_batches)}")
        
        retry = 0
        while retry < max_retries:
            try:
                if batch_idx > 0 or retry > 0:
                    time.sleep(2)
                
                # Download batch
                batch_df = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by='ticker',
                    threads=False,
                    show_errors=False
                )
                
                if not batch_df.empty:
                    for ticker in batch:
                        try:
                            if len(batch) > 1 and ticker in batch_df.columns.get_level_values(0):
                                data = batch_df[ticker]['Close']
                            elif len(batch) == 1:
                                data = batch_df['Close']
                            else:
                                failed.append(ticker)
                                continue
                            
                            if not data.empty and not data.isna().all():
                                all_data[ticker] = data
                            else:
                                failed.append(ticker)
                        except:
                            failed.append(ticker)
                    break
                else:
                    retry += 1
                    if retry >= max_retries:
                        failed.extend(batch)
                    
            except:
                retry += 1
                if retry >= max_retries:
                    failed.extend([t for t in batch if t not in all_data])
                time.sleep(2 * retry)
        
        progress_bar.progress((batch_idx + 1) / len(ticker_batches))
    
    progress_bar.empty()
    status_text.empty()
    
    # Try individual downloads for missing (max 5)
    missing = [t for t in tickers if t not in all_data]
    if 0 < len(missing) <= 5 and len(all_data) > 0:
        st.info(f"🔄 Retrying {len(missing)} individual tickers...")
        for ticker in missing:
            try:
                time.sleep(1.5)
                hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
                if not hist.empty and 'Close' in hist.columns and not hist['Close'].isna().all():
                    all_data[ticker] = hist['Close']
                    if ticker in failed:
                        failed.remove(ticker)
            except:
                if ticker not in failed:
                    failed.append(ticker)
    
    if not all_data:
        return None, list(set(failed))
    
    # Combine and resample
    combined = pd.DataFrame(all_data)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.resample('Q').last().dropna(how='all')
    
    if combined.empty:
        return None, list(set(failed))
    
    return combined, list(set(failed))

# Initialize demo mode state
if 'use_demo_mode' not in st.session_state:
    st.session_state.use_demo_mode = False

# Demo Mode
if st.session_state.use_demo_mode:
    st.warning("🎮 **Demo Mode Active** - Using simulated stock data")
    
    np.random.seed(42)
    quarters = pd.period_range(
        start=topics_df["Quarter_Date"].min(), 
        end=topics_df["Quarter_Date"].max(), 
        freq='Q'
    )
    
    stock_df = pd.DataFrame(index=quarters.to_timestamp(), columns=tickers[:12])
    
    for col in stock_df.columns:
        trend = np.random.choice([0.02, 0.01, -0.005])
        stock_df[col] = 100 * (1 + trend + np.random.randn(len(stock_df)) * 0.08).cumprod()
    
    valid_tickers = list(stock_df.columns)
    st.success(f"✅ Demo: {len(valid_tickers)} companies, {stock_df.shape[0]} quarters")
    st.info(f"📅 {stock_df.index.min().strftime('%Y-%m-%d')} to {stock_df.index.max().strftime('%Y-%m-%d')}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Real Data"):
            st.session_state.use_demo_mode = False
            st.rerun()
    with col2:
        if st.button("🎲 Regenerate Demo"):
            st.cache_data.clear()
            st.rerun()

else:
    # Real data fetching
    st.info("📊 Fetching stock data from Yahoo Finance...")
    st.write("⏱️ This may take 30-90 seconds. Please wait...")
    
    start_date = pd.Timestamp(topics_df["Quarter_Date"].min())
    end_date = pd.Timestamp(topics_df["Quarter_Date"].max()) + pd.DateOffset(months=3)
    
    try:
        stock_df, failed_tickers = fetch_stock_data_robust(tickers, start_date, end_date)
        
        # Check if we have enough data
        if stock_df is None or stock_df.empty or len(stock_df.columns) < 5:
            st.error("❌ Insufficient stock data")
            
            with st.expander("📋 Troubleshooting"):
                st.write("""
                **Common issues:**
                - Yahoo Finance rate limiting (Streamlit Cloud shared IP)
                - Network timeouts
                - Invalid date range
                
                **Solutions:**
                1. **Wait 5 minutes** then click Retry
                2. Try during **off-peak hours** (6-9 AM IST)
                3. Use **Demo Mode** (works perfectly!)
                """)
                if failed_tickers:
                    st.write(f"**Failed:** {', '.join(failed_tickers[:10])}")
            
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
        
        # Success!
        valid_tickers = list(stock_df.columns)
        success_rate = (len(valid_tickers) / len(tickers)) * 100
        
        if success_rate >= 70:
            st.success(f"✅ Success: {len(valid_tickers)}/{len(tickers)} companies ({success_rate:.0f}%)")
        else:
            st.warning(f"⚠️ Partial: {len(valid_tickers)}/{len(tickers)} companies ({success_rate:.0f}%)")
        
        if failed_tickers:
            with st.expander(f"⚠️ Failed: {len(failed_tickers)} tickers"):
                st.write(", ".join(sorted(failed_tickers)))
        
        st.info(f"📅 Data: {stock_df.index.min().strftime('%Y-%m-%d')} to {stock_df.index.max().strftime('%Y-%m-%d')}")
        
        if success_rate < 90:
            with st.expander("💡 Want complete data?"):
                st.write("Demo mode has 100% data coverage")
                if st.button("Switch to Demo"):
                    st.session_state.use_demo_mode = True
                    st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("🎮 Demo Mode", type="primary"):
                st.session_state.use_demo_mode = True
                st.rerun()
        st.stop()

# === ANALYSIS SECTION ===
st.markdown("---")

# Compute returns
returns_df = stock_df.pct_change().dropna(how="all")
returns_df.index = pd.to_datetime(returns_df.index)
returns_df["Quarter_Period"] = returns_df.index.to_period("Q")

# Sector returns
sector_returns = {}
for topic, comps in topic_to_companies.items():
    valid_comps = [c for c in comps if c in returns_df.columns]
    if valid_comps:
        sector_returns[topic] = returns_df[valid_comps].mean(axis=1)

sector_df = pd.DataFrame(sector_returns).dropna(how="all")
sector_df["Quarter_Period"] = returns_df["Quarter_Period"]

# Merge topics + stocks
st.markdown("## 🔗 Merging Topic and Stock Data")

topics_df["Quarter_Period"] = topics_df["Quarter_Date"].dt.to_period("Q")

merged_results = []
for topic, comps in topic_to_companies.items():
    if topic not in sector_df.columns:
        continue
    
    topic_rows = topics_df[topics_df["Topic_Label"] == topic]
    if topic_rows.empty:
        continue
    
    for _, row in topic_rows.iterrows():
        q_period = row["Quarter_Period"]
        matching = sector_df[sector_df["Quarter_Period"] == q_period]
        
        if not matching.empty:
            topic_rank = 6 - row["Rank"]
            sector_return = matching[topic].iloc[0]
            
            merged_results.append({
                "Quarter": row["Quarter"],
                "Quarter_Date": row["Quarter_Date"],
                "Topic": topic,
                "Topic_Strength": topic_rank,
                "Sector_Return": sector_return
            })

merged_df = pd.DataFrame(merged_results).dropna()

if merged_df.empty:
    st.error("❌ No overlapping quarters found")
    st.info("Check that topic dates and stock dates overlap")
    st.stop()

st.success(f"✅ Merged {len(merged_df)} data points across {merged_df['Quarter'].nunique()} quarters")

# Correlation Analysis
st.markdown("## 🔍 Topic–Sector Correlation")

corr_summary = (
    merged_df.groupby("Topic")[["Topic_Strength", "Sector_Return"]]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
    .rename(columns={"Sector_Return": "Correlation"})
    .drop(columns=["level_1"])
)

st.dataframe(
    corr_summary.style.format({"Correlation": "{:.2f}"}),
    use_container_width=True
)

fig = px.bar(
    corr_summary,
    x="Topic",
    y="Correlation",
    title="Correlation: Topic Strength vs Sector Returns",
    color="Correlation",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0
)
fig.update_layout(height=500, xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Temporal Analysis
st.markdown("## ⏳ Topic Strength vs Sector Performance")

selected_topic = st.selectbox(
    "Select Topic:",
    options=sorted(topic_to_companies.keys())
)

if selected_topic in merged_df["Topic"].unique():
    subset = merged_df[merged_df["Topic"] == selected_topic].sort_values("Quarter_Date")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=subset["Quarter_Date"],
        y=subset["Topic_Strength"],
        name="Topic Strength",
        mode="lines+markers",
        yaxis="y1",
        line=dict(color='#1f77b4', width=3)
    ))
    fig2.add_trace(go.Scatter(
        x=subset["Quarter_Date"],
        y=subset["Sector_Return"],
        name="Sector Return",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color='#ff7f0e', width=3)
    ))
    
    fig2.update_layout(
        title=f"{selected_topic}: Topic Strength vs Returns",
        xaxis_title="Quarter",
        yaxis=dict(title="Topic Strength", side="left"),
        yaxis2=dict(title="Sector Return (%)", side="right", overlaying="y"),
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📊 Data Table"):
        st.dataframe(
            subset[["Quarter", "Topic_Strength", "Sector_Return"]].style.format({
                "Topic_Strength": "{:.2f}",
                "Sector_Return": "{:.2%}"
            })
        )

# Export
st.markdown("## 💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    csv1 = merged_df.to_csv(index=False)
    st.download_button(
        "📥 Download Topic-Stock Data",
        data=csv1,
        file_name="topic_stock_correlation.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    csv2 = topics_df.to_csv(index=False)
    st.download_button(
        "📥 Download Quarterly Topics",
        data=csv2,
        file_name="quarterly_topics.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Topic–Stock Impact Analysis</div>",
    unsafe_allow_html=True
)