import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Regime Detection",
    page_icon="🚨",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#e74c3c;'>🚨 Market Regime Detection</h1>", unsafe_allow_html=True)
st.markdown("### Classifying market states and identifying rhetoric patterns")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

episode_file = st.sidebar.text_input(
    "Episode Topics CSV path",
    value="data/episode_topics.csv",
    help="Path to the file exported from Topic Modeling"
)

vol_window = st.sidebar.slider("Volatility Window (Days)", min_value=5, max_value=60, value=20)

# Load topic data
try:
    df_episodes = pd.read_csv(episode_file)
    df_episodes['date'] = pd.to_datetime(df_episodes['date'])
    st.success(f"✅ Loaded {len(df_episodes)} episodes.")
except Exception as e:
    st.error(f"Could not load episode file: {e}")
    st.info("💡 Please run Topic Modeling first and export the data to data/episode_topics.csv")
    st.stop()

# Fetch Market Data
st.info("📊 Fetching Nifty 50 and India VIX data...")

start_date = df_episodes['date'].min() - pd.DateOffset(months=3)
end_date = df_episodes['date'].max() + pd.DateOffset(days=5)

@st.cache_data
def get_market_data(start, end):
    # Nifty 50
    data_nifty = yf.download("^NSEI", start=start, end=end, progress=False)
    # India VIX
    data_vix = yf.download("^INDIAVIX", start=start, end=end, progress=False)
    
    # Robustly extract Price column
    def extract_price(df):
        if df.empty:
            return pd.Series(dtype='float64')
        # Handle MultiIndex columns (common in newer yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                return df['Adj Close'].iloc[:, 0]
            return df['Close'].iloc[:, 0]
        # Standard Index
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        return df['Close']
        
    nifty = extract_price(data_nifty)
    vix = extract_price(data_vix)
    
    return nifty, vix

nifty_data, vix_data = get_market_data(start_date, end_date)

if nifty_data.empty:
    st.error("Could not fetch Nifty 50 data.")
    st.stop()

# Process Market Data
market_df = pd.DataFrame(index=nifty_data.index)
market_df['Price'] = nifty_data
market_df['Returns'] = market_df['Price'].pct_change()
market_df['Volatility'] = market_df['Returns'].rolling(window=vol_window).std() * np.sqrt(252) * 100

if not vix_data.empty:
    market_df['VIX'] = vix_data
    # Forward fill VIX since it might have missing days
    market_df['VIX'] = market_df['VIX'].ffill()

# Regime Classification Logic
st.sidebar.subheader("Regime Thresholds (Percentiles)")
low_v = st.sidebar.slider("Stable Threshold (%)", 10, 50, 33)
high_v = st.sidebar.slider("Volatile Threshold (%)", 50, 90, 66)

# Use VIX if available, else Realized Volatility
regime_col = 'VIX' if 'VIX' in market_df.columns else 'Volatility'
v_metrics = market_df[regime_col].dropna()
low_thresh = np.percentile(v_metrics, low_v)
high_thresh = np.percentile(v_metrics, high_v)

def classify_regime(val):
    if pd.isna(val): return None
    if val < low_thresh: return "Stable"
    if val > high_thresh: return "Volatile"
    return "Transition"

market_df['Regime'] = market_df[regime_col].apply(classify_regime)

# Visualization 1: Market Price and Regimes
st.subheader("📈 Market Price and Regimes")

fig_regime = go.Figure()

# Add price line
fig_regime.add_trace(go.Scatter(
    x=market_df.index, y=market_df['Price'],
    name="Nifty 50", line=dict(color='black', width=1.5)
))

# Add regime highlights
colors = {"Stable": "rgba(46, 204, 113, 0.3)", "Transition": "rgba(241, 196, 15, 0.3)", "Volatile": "rgba(231, 76, 60, 0.3)"}

for regime in ["Stable", "Transition", "Volatile"]:
    mask = market_df['Regime'] == regime
    # Find contiguous blocks
    # Simple way to highlight in plotly is to add shapes or scatter with fill
    # Here we'll use colored background for simplicity in a dashboard
    pass

# Better visualization for regimes: Scatter plot with colored background
fig_regime = px.line(market_df, x=market_df.index, y="Price", title="Nifty 50 Index with Market Regimes")

# Add shapes for regimes
for i in range(len(market_df)-1):
    regime = market_df['Regime'].iloc[i]
    if regime:
        fig_regime.add_vrect(
            x0=market_df.index[i], x1=market_df.index[i+1],
            fillcolor=colors[regime], opacity=0.5,
            layer="below", line_width=0,
        )

st.plotly_chart(fig_regime, use_container_width=True)

# Align Speeches to Regimes
st.subheader("🎙️ Topic Prevalence by Market Regime")

# Merge speech dates with market regimes
df_merged = df_episodes.copy()
df_merged['Regime'] = df_merged['date'].apply(lambda x: market_df.loc[market_df.index.asof(x), 'Regime'] if x in market_df.index or market_df.index.asof(x) else None)

# Handle cases where speech date is not a trading day (asof handled most, but let's be sure)
df_merged = df_merged.dropna(subset=['Regime'])

topic_cols = [c for c in df_merged.columns if c not in ['episode', 'date', 'quarter', 'Regime']]

# Calculate mean topic intensity per regime
regime_analysis = df_merged.groupby('Regime')[topic_cols].mean().reset_index()

# Melt for visualization
regime_melted = regime_analysis.melt(id_vars='Regime', var_name='Topic', value_name='Intensity')

fig_bar = px.bar(
    regime_melted, x="Topic", y="Intensity", color="Regime",
    barmode="group", title="Average Topic Intensity by Market State",
    color_discrete_map={"Stable": "#2ecc71", "Transition": "#f1c40f", "Volatile": "#e74c3c"}
)
fig_bar.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_bar, use_container_width=True)

# Detailed Stats
col1, col2 = st.columns(2)

with col1:
    st.write("#### Regime Distribution in Speeches")
    regime_counts = df_merged['Regime'].value_counts()
    fig_pie = px.pie(values=regime_counts.values, names=regime_counts.index, 
                     color=regime_counts.index,
                     color_discrete_map={"Stable": "#2ecc71", "Transition": "#f1c40f", "Volatile": "#e74c3c"})
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.write("#### Dominant Topic per Regime")
    dominant_topics = regime_analysis.set_index('Regime').idxmax(axis=1)
    for regime, topic in dominant_topics.items():
        st.info(f"**{regime}**: {topic}")

# Temporal Shift Analysis
st.subheader("⏳ Regime Transitions vs Topic Spikes")

selected_topics = st.multiselect("Select Topics to Compare:", topic_cols, default=topic_cols[:2])

if selected_topics:
    fig_trend = go.Figure()
    
    # Add VIX or Volatility as background
    fig_trend.add_trace(go.Scatter(
        x=market_df.index, y=market_df[regime_col],
        name=f"Market {regime_col}", line=dict(color='rgba(150, 150, 150, 0.5)', dash='dot'),
        yaxis="y2"
    ))
    
    for topic in selected_topics:
        fig_trend.add_trace(go.Scatter(
            x=df_merged['date'], y=df_merged[topic],
            name=topic, mode='lines+markers'
        ))
        
    fig_trend.update_layout(
        title="Topic Intensity Spikes vs Market Volatility",
        xaxis_title="Date",
        yaxis=dict(title="Topic Intensity"),
        yaxis2=dict(title=regime_col, overlaying='y', side='right'),
        height=500
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")
st.markdown("💡 **Interpretation**: High intensity in 'Healthcare & Pandemic' topics often correlates with high-volatility regimes (e.g., 2020 Q1-Q2).")
