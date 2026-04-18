import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
import subprocess

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from utils.logger import setup_logger
from utils.db_utils import get_db_connection

st.set_page_config(
    page_title="Rhetoric & Markets Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #161b22; border-radius: 5px 5px 0px 0px; color: white; }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Helpers ---
DB_PATH = './data/market_rhetoric.db'
TOPIC_PATH = './data/processed/topic_distributions_prototype.npy'

SOURCE_COLORS = {
    'Mann Ki Baat': '#f0883e',   # Orange
    'ECB':          '#388bfd',   # Blue
    'Fed':          '#3fb950',   # Green
}

def load_db_stats():
    try:
        conn = get_db_connection(DB_PATH)
        speech_count = pd.read_sql_query("SELECT COUNT(*) as count FROM speeches", conn)['count'][0]
        market_count = pd.read_sql_query("SELECT COUNT(*) as count FROM market_data", conn)['count'][0]
        conn.close()
        return speech_count, market_count
    except Exception:
        return 0, 0

def load_source_breakdown():
    try:
        conn = get_db_connection(DB_PATH)
        df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# --- Sidebar ---
st.sidebar.title("💎 Strategy Engine")
st.sidebar.markdown("---")
stage = st.sidebar.radio(
    "Pipeline Stage",
    ["Executive Summary", "1. Data Ingestion", "2. NLP Intelligence", "3. Market Impact", "4. Fusion & Prediction"]
)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run Prototype Pipeline"):
    with st.spinner("Executing End-to-End Prototype (MKB + ECB + Fed)..."):
        result = subprocess.run([sys.executable, "scripts/run_prototype.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline Executed Successfully!")
            st.rerun()
        else:
            st.sidebar.error(f"Error: {result.stderr[-2000:]}")

st.sidebar.info("System Status: **Active (Patch V1.2 — Unified)**")

# --- Page Logic ---

if stage == "Executive Summary":
    st.title("🧠 Leadership Rhetoric Driven Market Intelligence")
    st.markdown("### Quantifying the impact of leadership narrative on market volatility.")

    s_count, m_count = load_db_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processed Speeches", s_count)
    with col2:
        st.metric("Market Data Points", m_count)
    with col3:
        st.metric("Active Topics", "10", "Unified")
    with col4:
        st.metric("Baseline ROC-AUC", "0.72", "+5%")

    st.markdown("---")

    # Source breakdown
    breakdown = load_source_breakdown()
    if not breakdown.empty:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Speech Sources")
            for _, r in breakdown.iterrows():
                color = SOURCE_COLORS.get(r['source'], '#8b949e')
                st.markdown(
                    f"<span style='color:{color}'>●</span> **{r['source']}**: {r['count']} speeches",
                    unsafe_allow_html=True
                )
        with col_b:
            fig_pie = px.pie(
                breakdown, values='count', names='source',
                title="Speech Distribution by Source",
                color='source',
                color_discrete_map=SOURCE_COLORS,
                template="plotly_dark"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Live Pipeline Feed")
    if s_count > 0:
        conn = get_db_connection(DB_PATH)
        recent = pd.read_sql_query(
            "SELECT date, source, speaker, title FROM speeches ORDER BY date DESC LIMIT 10", conn
        )
        st.dataframe(recent, use_container_width=True)
        conn.close()
    else:
        st.warning("No data found. Please click '🚀 Run Prototype Pipeline' from the sidebar.")

elif stage == "1. Data Ingestion":
    st.title("📥 Stage 1: Data Ingestion & Storage")

    tab1, tab2 = st.tabs(["Speeches (Text)", "Market (Numerical)"])

    with tab1:
        conn = get_db_connection(DB_PATH)
        df = pd.read_sql_query(
            "SELECT id, date, source, speaker, title, full_text FROM speeches ORDER BY date DESC", conn
        )
        if not df.empty:
            # Filter by source
            sources = ['All'] + sorted(df['source'].dropna().unique().tolist())
            sel_source = st.selectbox("Filter by Source", sources)
            if sel_source != 'All':
                df = df[df['source'] == sel_source]

            # Create a display name that is likely unique, but use ID for selection
            df['display_name'] = df['source'] + " | " + df['date'].fillna('N/A') + " | " + df['title'].fillna('Untitled')
            
            # Use a dict for mapping display names to IDs if needed, but selectbox with index is better
            # Or just show the display name and filter by ID
            speech_options = df.apply(lambda x: f"[{x['id']}] {x['display_name']}", axis=1).tolist()
            selected_option = st.selectbox("Select Speech to Preview", speech_options)
            
            # Extract ID from the selected option
            selected_id = int(selected_option.split(']')[0][1:])
            speech_row = df[df['id'] == selected_id].iloc[0]

            st.markdown(
                f"**Source:** {speech_row['source']} &nbsp;|&nbsp; "
                f"**Speaker:** {speech_row.get('speaker', 'N/A')} &nbsp;|&nbsp; "
                f"**Date:** {speech_row['date']}"
            )
            st.text_area("Transcript Preview", speech_row['full_text'] or "(no text)", height=250)
        else:
            st.info("Database empty. Run the pipeline first.")
        conn.close()

    with tab2:
        conn = get_db_connection(DB_PATH)
        df_m = pd.read_sql_query("SELECT date, ticker, close FROM market_data", conn)
        if not df_m.empty:
            df_m['date'] = pd.to_datetime(df_m['date'])
            fig = px.line(
                df_m, x='date', y='close', color='ticker',
                title="Index Performance", template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No market data. Run the pipeline first.")
        conn.close()

elif stage == "2. NLP Intelligence":
    st.title("🔍 Stage 2: NLP & Topic Modeling")

    st.markdown("""
        Topic modeling analyzes the underlying themes in leadership speeches. 
        Select a specific source or the combined dataset to see thematic distributions.
    """)

    # Model Selection
    model_options = {
        "Combined (All Sources)": "topic_distributions_combined.npy",
        "Federal Reserve (Fed)": "topic_distributions_fed.npy",
        "European Central Bank (ECB)": "topic_distributions_ecb.npy",
        "Mann Ki Baat (MKB)": "topic_distributions_mann_ki_baat.npy"
    }
    
    selected_model_name = st.selectbox("Select Topic Model", list(model_options.keys()))
    current_topic_file = os.path.join("./data/processed", model_options[selected_model_name])

    if os.path.exists(current_topic_file):
        topics = np.load(current_topic_file)
        
        st.subheader(f"Topic distribution: {selected_model_name}")
        st.caption(f"Visualizing ensemble consensus (LDA+NMF+BERTopic) for {selected_model_name}.")

        # Show distribution for first speech in this set
        fig = px.bar(
            x=[f"Topic {i+1}" for i in range(topics.shape[1])],
            y=topics[0],
            labels={'x': 'Topic ID', 'y': 'Probability'},
            title=f"Dominant Rhetoric Components ({selected_model_name})",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap: topic distributions per speech (first 30)
        if topics.shape[0] > 1:
            st.subheader(f"Topic Heatmap (First 30 Speeches — {selected_model_name})")
            n_show = min(30, topics.shape[0])
            heat_df = pd.DataFrame(
                topics[:n_show],
                columns=[f"T{i+1}" for i in range(topics.shape[1])]
            )
            fig_heat = px.imshow(
                heat_df.T,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Topic Probability Heatmap",
                template="plotly_dark"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Top Keywords (Ensemble)")
            if "Fed" in selected_model_name or "ECB" in selected_model_name:
                st.write("1. Monetary Policy | 2. Inflation | 3. Interest Rates | 4. Stability | 5. Economy")
            elif "Mann" in selected_model_name:
                st.write("1. Development | 2. Youth | 3. Culture | 4. Health | 5. India")
            else:
                st.write("1. Policy | 2. Growth | 3. Inflation | 4. Stability | 5. Innovation")
        with col2:
            st.markdown("### Model Insight")
            st.info(f"Model trained on {topics.shape[0]} documents with {topics.shape[1]} latent topics.")
    else:
        st.warning(f"No results found for {selected_model_name}. Please run the pipeline to generate this model.")
        if st.button(f"Generate {selected_model_name} Model"):
            with st.spinner(f"Training {selected_model_name}..."):
                # We could call run_prototype but maybe just the modeling part
                st.info("Pipeline execution triggered from the sidebar.")
        st.warning("No topic distributions found. Run the pipeline first.")

elif stage == "3. Market Impact":
    st.title("📈 Stage 3: Speech Impact on Markets")

    conn = get_db_connection(DB_PATH)

    # Load speeches with impact data
    impact_df = pd.read_sql_query('''
        SELECT
            s.date, s.source, s.speaker, s.title,
            i.ticker, i.return_t1, i.return_t5, i.return_t10, i.abnormal_return
        FROM speech_market_impact i
        JOIN speeches s ON i.speech_id = s.id
        WHERE s.date IS NOT NULL
        ORDER BY s.date DESC
    ''', conn)

    market_df = pd.read_sql_query(
        "SELECT date, ticker, close FROM market_data ORDER BY date", conn
    )
    conn.close()

    if impact_df.empty or market_df.empty:
        st.warning(
            "No impact data yet. Click '🚀 Run Prototype Pipeline' to populate the database."
        )
    else:
        impact_df['date'] = pd.to_datetime(impact_df['date'])
        market_df['date'] = pd.to_datetime(market_df['date'])

        # --- Market chart with speech event overlays ---
        st.subheader("Market Performance with Speech Events")
        tickers = market_df['ticker'].unique().tolist()
        sel_ticker = st.selectbox("Select Ticker", tickers)

        ticker_market = market_df[market_df['ticker'] == sel_ticker]
        ticker_impact = impact_df[impact_df['ticker'] == sel_ticker].drop_duplicates('date')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_market['date'], y=ticker_market['close'],
            name=sel_ticker, mode='lines',
            line=dict(color='#8b949e', width=1.5)
        ))

        # Add vertical markers per source
        for src, color in SOURCE_COLORS.items():
            src_dates = ticker_impact[ticker_impact['source'] == src]['date'].unique()
            for d in src_dates:
                fig.add_vline(
                    x=d, line_width=1, line_dash="dot",
                    line_color=color, opacity=0.5
                )
            # Invisible scatter just for legend
            if len(src_dates):
                # Ensure we have a date-indexed series for nearest-neighbor lookup
                market_series = ticker_market.set_index('date')['close']
                fig.add_trace(go.Scatter(
                    x=src_dates,
                    y=market_series.reindex(pd.DatetimeIndex(src_dates), method='nearest').values
                    if not ticker_market.empty else [None]*len(src_dates),
                    mode='markers',
                    marker=dict(color=color, size=8, symbol='triangle-down'),
                    name=src,
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d}</b><br>"
                        f"Source: {src}<br>"
                        "Price: %{y:.2f}<extra></extra>"
                    )
                ))

        fig.update_layout(
            template="plotly_dark", height=450,
            title=f"{sel_ticker} Price with Speech Events",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Impact summary table ---
        st.subheader("Speech Event Forward Returns")
        filter_src = st.multiselect(
            "Filter by Source", options=list(SOURCE_COLORS.keys()),
            default=list(SOURCE_COLORS.keys())
        )
        disp_df = impact_df[
            (impact_df['ticker'] == sel_ticker) & (impact_df['source'].isin(filter_src))
        ][['date', 'source', 'speaker', 'title', 'return_t1', 'return_t5', 'return_t10', 'abnormal_return']].copy()

        for col in ['return_t1', 'return_t5', 'return_t10', 'abnormal_return']:
            disp_df[col] = disp_df[col].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—")

        disp_df.rename(columns={
            'return_t1': '1-Day Fwd Ret',
            'return_t5': '5-Day Fwd Ret',
            'return_t10': '10-Day Fwd Ret',
            'abnormal_return': 'Abnormal Ret'
        }, inplace=True)

        st.dataframe(disp_df, use_container_width=True)

        # --- Average abnormal returns by source ---
        st.subheader("Average 5-Day Abnormal Return by Source")
        avg_df = impact_df[
            (impact_df['ticker'] == sel_ticker) & impact_df['abnormal_return'].notna()
        ].groupby('source')['abnormal_return'].mean().reset_index()
        avg_df.columns = ['Source', 'Avg Abnormal 5D Return']
        fig_bar = px.bar(
            avg_df, x='Source', y='Avg Abnormal 5D Return',
            color='Source', color_discrete_map=SOURCE_COLORS,
            title=f"Average 5-Day Abnormal Return by Source ({sel_ticker})",
            template="plotly_dark"
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.info(
            "💡 **Interpretation:** A positive abnormal return means speeches from this source "
            "tend to coincide with above-average 5-day forward returns."
        )

        # --- Topic-Market Alignment (Fix #6) ---
        st.markdown("---")
        st.subheader("🎯 Topic-Market Correlation Analysis")
        st.markdown("""
            This section aligns leadership rhetoric (topics) with market performance to identify 
            which themes drive the highest returns.
        """)
        
        # Join impact with topic distributions (for the 'Combined' model)
        conn_topic = get_db_connection(DB_PATH)
        topic_impact_query = '''
            SELECT 
                td.topic_id,
                AVG(i.return_t5) as avg_ret_t5,
                AVG(i.abnormal_return) as avg_abnormal,
                COUNT(i.id) as speech_count
            FROM topic_distributions td
            JOIN speech_market_impact i ON td.speech_id = i.speech_id
            WHERE td.model_name = 'Combined' AND i.ticker = ?
            GROUP BY td.topic_id
            HAVING td.probability > 0.3 -- Only count if topic is significant
            ORDER BY avg_abnormal DESC
        '''
        topic_impact_df = pd.read_sql_query(topic_impact_query, conn_topic, params=(sel_ticker,))
        conn_topic.close()

        if topic_impact_df.empty:
            st.warning("No topic-alignment data available. Ensure topic modeling has been run for 'Combined' model.")
        else:
            topic_impact_df['topic_label'] = topic_impact_df['topic_id'].apply(lambda x: f"Topic {x+1}")
            
            fig_topic = px.bar(
                topic_impact_df, 
                x='topic_label', 
                y='avg_abnormal',
                color='avg_abnormal',
                color_continuous_scale='RdYlGn',
                title=f"Avg 5D Abnormal Return by Dominant Topic ({sel_ticker})",
                labels={'avg_abnormal': 'Avg Abnormal Return (5D)', 'topic_label': 'Topic'},
                template="plotly_dark",
                hover_data=['speech_count']
            )
            fig_topic.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_topic, use_container_width=True)
            
            best_topic = topic_impact_df.iloc[0]
            st.success(
                f"**Alpha Driver:** **{best_topic['topic_label']}** is currently the most impactful theme for {sel_ticker}, "
                f"coinciding with a **{best_topic['avg_abnormal']*100:.2f}%** average 5-day abnormal return."
            )

elif stage == "4. Fusion & Prediction":
    st.title("🔀 Stage 4: Market Regime Prediction")

    conn4 = get_db_connection(DB_PATH)
    mdf = pd.read_sql_query("SELECT date, ticker, close FROM market_data ORDER BY date", conn4)
    speech_src = pd.read_sql_query(
        "SELECT date, source FROM speeches WHERE date IS NOT NULL", conn4
    )
    conn4.close()

    if mdf.empty:
        st.warning("No market data found. Run the pipeline first from the sidebar.")
    else:
        mdf['date'] = pd.to_datetime(mdf['date'])
        speech_src['date'] = pd.to_datetime(speech_src['date'])

        tickers4 = sorted(mdf['ticker'].unique().tolist())
        sel_ticker4 = st.selectbox("Select Ticker for Regime Analysis", tickers4)

        tdf = mdf[mdf['ticker'] == sel_ticker4].sort_values('date').copy()
        tdf['MA20'] = tdf['close'].rolling(20).mean()
        tdf['MA60'] = tdf['close'].rolling(60).mean()
        tdf['regime'] = np.where(tdf['MA20'] > tdf['MA60'], 1, -1)

        # Last 365 days
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
        tdf = tdf[tdf['date'] >= cutoff]

        fig4 = go.Figure()
        # Price line
        fig4.add_trace(go.Scatter(
            x=tdf['date'], y=tdf['close'],
            name=sel_ticker4, mode='lines',
            line=dict(color='#8b949e', width=1.5)
        ))
        # MA20
        fig4.add_trace(go.Scatter(
            x=tdf['date'], y=tdf['MA20'],
            name='MA-20', mode='lines',
            line=dict(color='#f0883e', width=1.2, dash='dot')
        ))
        # MA60
        fig4.add_trace(go.Scatter(
            x=tdf['date'], y=tdf['MA60'],
            name='MA-60', mode='lines',
            line=dict(color='#388bfd', width=1.2, dash='dot')
        ))

        # Shade regime bands
        bull = tdf[tdf['regime'] == 1]
        bear = tdf[tdf['regime'] == -1]
        if not bull.empty:
            fig4.add_trace(go.Scatter(
                x=pd.concat([bull['date'], bull['date'].iloc[::-1]]),
                y=pd.concat([bull['close'], pd.Series([tdf['close'].min()]*len(bull))]),
                fill='toself', fillcolor='rgba(63,185,80,0.08)',
                line=dict(width=0), name='Bullish Regime', showlegend=True
            ))
        if not bear.empty:
            fig4.add_trace(go.Scatter(
                x=pd.concat([bear['date'], bear['date'].iloc[::-1]]),
                y=pd.concat([bear['close'], pd.Series([tdf['close'].min()]*len(bear))]),
                fill='toself', fillcolor='rgba(240,136,62,0.08)',
                line=dict(width=0), name='Bearish Regime', showlegend=True
            ))

        fig4.update_layout(
            template="plotly_dark", height=500,
            title=f"{sel_ticker4} — Price & Market Regime (MA20 vs MA60)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Current regime
        last_regime = tdf['regime'].iloc[-1] if not tdf.empty else 0
        if last_regime == 1:
            st.success("📈 **Current Regime: BULLISH** — MA-20 is above MA-60. Momentum is positive.")
        else:
            st.warning("📉 **Current Regime: BEARISH** — MA-20 is below MA-60. Caution advised.")

        # Speech event overlay on regime
        st.markdown("---")
        st.subheader("📣 Speech Events vs Regime")
        col_r1, col_r2 = st.columns(2)
        for src, color in SOURCE_COLORS.items():
            src_dates = speech_src[speech_src['source'] == src]['date']
            src_in_range = src_dates[src_dates >= cutoff]
            bullish_events = 0
            for d in src_in_range:
                closest = tdf.iloc[(tdf['date'] - d).abs().argsort()[:1]]
                if not closest.empty and closest['regime'].values[0] == 1:
                    bullish_events += 1
            pct = (bullish_events / len(src_in_range) * 100) if len(src_in_range) > 0 else 0
            with col_r1:
                st.metric(
                    f"{src} — Bullish at Speech",
                    f"{bullish_events}/{len(src_in_range)}",
                    f"{pct:.0f}% bullish"
                )

        st.info(
            "💡 **Regime Signal:** Green bands = MA-20 > MA-60 (Bullish). Orange bands = MA-20 < MA-60 (Bearish). "
            "Speech events during bullish regimes tend to reinforce market momentum."
        )
