import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="Predictive Analysis",
    page_icon="🔮",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#9b59b6;'>🔮 Early Warning System</h1>", unsafe_allow_html=True)
st.markdown("### Predicting market regime shifts using leadership rhetoric")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Settings")

episode_file = st.sidebar.text_input(
    "Episode Topics CSV path",
    value="data/episode_topics.csv"
)

prediction_horizon = st.sidebar.slider("Prediction Horizon (Days Ahead)", 5, 60, 20)

# Load data
try:
    df_episodes = pd.read_csv(episode_file)
    df_episodes['date'] = pd.to_datetime(df_episodes['date'])
    topic_cols = [c for c in df_episodes.columns if c not in ['episode', 'date', 'quarter']]
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# Market Data Fetching (Reusing logic from Regime page)
@st.cache_data
def get_market_metrics(start, end):
    data_nifty = yf.download("^NSEI", start=start, end=end, progress=False)
    data_vix = yf.download("^INDIAVIX", start=start, end=end, progress=False)
    
    def extract_price(df):
        if df.empty:
            return pd.Series(dtype='float64')
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.get_level_values(0):
                return df['Adj Close'].iloc[:, 0]
            return df['Close'].iloc[:, 0]
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        return df['Close']

    nifty = extract_price(data_nifty)
    vix = extract_price(data_vix)
    
    m_df = pd.DataFrame(index=nifty.index)
    m_df['Price'] = nifty
    m_df['Returns'] = m_df['Price'].pct_change()
    m_df['Volatility'] = m_df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
    if not vix.empty:
        m_df['VIX'] = vix.ffill()
    
    # Define Regimes
    v_col = 'VIX' if 'VIX' in m_df.columns else 'Volatility'
    metrics_subset = m_df[v_col].dropna()
    if not metrics_subset.empty:
        low_t = np.percentile(metrics_subset, 33)
        high_t = np.percentile(metrics_subset, 66)
    else:
        low_t, high_t = 20, 40 # Fallbacks
    
    def get_regime(x):
        if pd.isna(x): return None
        if x < low_t: return 0 # Stable
        if x > high_t: return 2 # Volatile
        return 1 # Transition
        
    m_df['Regime'] = m_df[v_col].apply(get_regime)
    return m_df

start_date = df_episodes['date'].min() - pd.DateOffset(months=3)
end_date = df_episodes['date'].max() + pd.DateOffset(days=5)
market_df = get_market_metrics(start_date, end_date)

# Feature Engineering: Lagged Topics
st.subheader("🛠️ Feature Engineering & Lead-Lag Analysis")

# Create a daily topic dataframe by reindexing and filling
daily_topics = df_episodes.set_index('date')[topic_cols].reindex(market_df.index).ffill(limit=30)
# Only keep days where we actually had a speech recently (within 30 days)
daily_topics = daily_topics.fillna(0)

# Target: Future Regime
market_df['Target_Regime'] = market_df['Regime'].shift(-prediction_horizon)

# Combine
model_df = pd.concat([daily_topics, market_df[['Target_Regime']]], axis=1).dropna()

st.write(f"Model prepared with **{len(model_df)}** training samples.")

# Prediction horizon visualization
col1, col2 = st.columns(2)
with col1:
    st.info(f"Predicting if the market will be **Stable**, **Transition**, or **Volatile** in **{prediction_horizon} days** based on current speech topics.")

# Model Training
st.subheader("🤖 Predictive Model (Random Forest)")

if st.button("🚀 Train Predictive Model"):
    X = model_df[topic_cols]
    y = model_df['Target_Regime']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Metrics
    score = clf.score(X_test, y_test)
    st.success(f"Model Accuracy: {score:.2%}")
    
    # Feature Importance
    importances = pd.DataFrame({
        'Topic': topic_cols,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_imp = px.bar(importances, x="Importance", y="Topic", orientation='h', 
                     title="Predictive Power of Topics (Feature Importance)")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Warnings
    st.subheader("🚨 Early Warning Indicators")
    top_predictors = importances.head(3)['Topic'].tolist()
    st.write("The following topics have the highest influence on future market regimes:")
    for t in top_predictors:
        st.warning(f"**{t}**")

# Lead-Lag Visualization
st.subheader("📉 Lead-Lag Correlation")

selected_topic = st.selectbox("Select Topic for Cross-Correlation:", topic_cols)

# Simple cross-correlation
lags = range(-30, 31)
corrs = []
v_col = 'VIX' if 'VIX' in market_df.columns else 'Volatility'

for l in lags:
    c = daily_topics[selected_topic].corr(market_df[v_col].shift(-l))
    corrs.append(c)

fig_corr = px.line(x=list(lags), y=corrs, title=f"Cross-Correlation: {selected_topic} vs Market Volatility",
                   labels={'x': 'Lag (Days)', 'y': 'Correlation'})
fig_corr.add_vline(x=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig_corr, use_container_width=True)

st.info("💡 A peak at **negative lags** suggests the topic *precedes* market volatility (Early Warning). A peak at **positive lags** suggests the topic *responds* to the market.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Built with Streamlit | Early Warning Predictive System</div>", unsafe_allow_html=True)
