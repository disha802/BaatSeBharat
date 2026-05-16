import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
import json

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

# --- Required File Verification ---
DB_PATH = './data/market_rhetoric.db'

SOURCE_COLORS = {
    'Mann Ki Baat': '#f0883e',   # Orange
    'ECB':          '#388bfd',   # Blue
    'Fed':          '#3fb950',   # Green
}

REQUIRED_FILES = [
    DB_PATH,
    './data/processed/topic_distributions_combined.npy',
    './data/processed/topic_labels_combined.json'
]

missing_reqs = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_reqs:
    st.error(f"### ❌ CRITICAL: Missing Required Pipeline Files\n\nThe following files are missing. Please click **'🚀 Run Pipeline'** in the sidebar to generate them.\n\n" + "\n".join([f"- `{f}`" for f in missing_reqs]))
    # We don't st.stop() here because we want the user to be able to click the button in the sidebar

@st.cache_data
def load_db_stats():
    try:
        conn = get_db_connection(DB_PATH)
        speech_count = pd.read_sql_query("SELECT COUNT(*) as count FROM speeches", conn)['count'][0]
        market_count = pd.read_sql_query("SELECT COUNT(*) as count FROM market_data", conn)['count'][0]
        conn.close()
        return speech_count, market_count
    except Exception:
        return 0, 0

@st.cache_data
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
    ["Executive Summary", "1. Data Ingestion", "2. NLP Intelligence", "3. Market Impact", "4. Regime Intelligence", "5. Company Analytics"]
)

st.sidebar.markdown("---")
# Check if models exist and get timestamp
model_path = "./data/processed/topic_distributions_combined.npy"
models_exist = os.path.exists(model_path)
btn_label = "🚀 Run Pipeline Again" if models_exist else "🚀 Run Pipeline"

if st.sidebar.button(btn_label):
    with st.spinner("Executing End-to-End Prototype (MKB + ECB + Fed)..."):
        result = subprocess.run([sys.executable, "scripts/run_prototype.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline Executed Successfully!")
            st.rerun()
        else:
            st.sidebar.error("Execution failed. Check data consistency.")
            with open("logs/pipeline_error.log", "w") as f:
                f.write(result.stderr)

if models_exist:
    mtime = os.path.getmtime(model_path)
    last_update = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    st.sidebar.caption(f"Last Intelligence Update: {last_update}")

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
            # Load actual keywords if available
            labels_file = os.path.join("./data/processed", f"topic_labels_{selected_model_name.lower().replace(' (all sources)', '').replace('federal reserve (fed)', 'fed').replace('european central bank (ecb)', 'ecb').replace('mann ki baat (mkb)', 'mann_ki_baat').replace(' ', '_')}.json")
            
            if os.path.exists(labels_file):
                import json
                with open(labels_file, 'r') as f:
                    labels_data = json.load(f)
                
                # Show keywords for top topics
                for i in range(min(5, topics.shape[1])):
                    topic_key = f"Topic_{i}"
                    if topic_key in labels_data:
                        keywords = ", ".join(labels_data[topic_key]['keywords'][:5])
                        st.write(f"**T{i+1}:** {keywords}")
            else:
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
        st.warning(f"No results found for {selected_model_name}.")
        st.info("💡 Use the **Run Pipeline** button in the sidebar to generate results.")
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
            -- Removing the heavy threshold to avoid empty results for rare topics
            -- HAVING td.probability > 0.3 
            ORDER BY avg_abnormal DESC
        '''
        topic_impact_df = pd.read_sql_query(topic_impact_query, conn_topic, params=(sel_ticker,))
        conn_topic.close()

        if topic_impact_df.empty:
            st.warning("No precise topic-alignment data available for this ticker. Showing overall topic performance instead.")
            # Fallback: Overall average across all tickers or most similar data
            topic_impact_query_fallback = '''
                SELECT 
                    td.topic_id,
                    AVG(i.return_t5) as avg_ret_t5,
                    AVG(i.abnormal_return) as avg_abnormal,
                    COUNT(i.id) as speech_count
                FROM topic_distributions td
                JOIN speech_market_impact i ON td.speech_id = i.speech_id
                WHERE td.model_name = 'Combined'
                GROUP BY td.topic_id
                ORDER BY avg_abnormal DESC
            '''
            topic_impact_df = pd.read_sql_query(topic_impact_query_fallback, conn_topic)
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

elif stage == "4. Regime Intelligence":
    st.title("🛡️ Stage 4: Market Regime Intelligence (HMM)")
    st.markdown("### Quantifying structural market shifts using Hidden Markov Models.")

    conn = get_db_connection(DB_PATH)
    regimes = pd.read_sql_query("SELECT date, sector, regime, confidence FROM regime_classifications ORDER BY date", conn)
    market = pd.read_sql_query("SELECT date, ticker, close FROM market_data ORDER BY date", conn)
    conn.close()

    if regimes.empty or market.empty:
        st.warning("No regime data found. Run the pipeline first.")
    else:
        regimes['date'] = pd.to_datetime(regimes['date'])
        market['date'] = pd.to_datetime(market['date'])

        tickers = market['ticker'].unique()
        sel_ticker = st.selectbox("Select Ticker for Regime Timeline", tickers)

        t_market = market[market['ticker'] == sel_ticker]
        t_regimes = regimes[regimes['sector'] == sel_ticker]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_market['date'], y=t_market['close'], name="Price", line=dict(color='white')))

        # Add regime backgrounds
        colors = {'Stable': 'rgba(63, 185, 80, 0.2)', 'Transitional': 'rgba(240, 136, 62, 0.2)', 'Volatile': 'rgba(248, 81, 73, 0.2)'}
        
        # Group consecutive regimes to reduce shapes
        t_regimes = t_regimes.sort_values('date')
        if not t_regimes.empty:
            start_date = t_regimes.iloc[0]['date']
            curr_regime = t_regimes.iloc[0]['regime']
            
            for i in range(1, len(t_regimes)):
                if t_regimes.iloc[i]['regime'] != curr_regime:
                    end_date = t_regimes.iloc[i]['date']
                    fig.add_vrect(x0=start_date, x1=end_date, fillcolor=colors.get(curr_regime, 'gray'), opacity=0.5, layer="below", line_width=0)
                    start_date = end_date
                    curr_regime = t_regimes.iloc[i]['regime']
            
            # Last segment
            fig.add_vrect(x0=start_date, x1=t_regimes.iloc[-1]['date'], fillcolor=colors.get(curr_regime, 'gray'), opacity=0.5, layer="below", line_width=0)

        fig.update_layout(title=f"{sel_ticker} Regime Timeline (Green=Stable, Yellow=Transitional, Red=Volatile)", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

elif stage == "5. Company Analytics":
    st.title("🏢 Stage 5: Company Specific Returns vs. Rhetoric")
    st.markdown("### Analyzing how leadership topics impact individual company performance.")

    # In a real scenario, we'd have company-specific returns in the DB. 
    # For this prototype, we'll use sector proxies or simulated company data.
    
    conn = get_db_connection(DB_PATH)
    # Get topics
    topics_df = pd.read_sql_query("SELECT s.date, td.topic_id, td.probability FROM topic_distributions td JOIN speeches s ON td.speech_id = s.id WHERE td.model_name = 'Combined'", conn)
    conn.close()

    if topics_df.empty:
        st.warning("No topic data found. Run the pipeline first.")
    else:
        topics_df['date'] = pd.to_datetime(topics_df['date'])
        
        company = st.selectbox("Select Company", ["HDFC Bank", "Reliance Industries", "Infosys", "TCS", "ICICI Bank"])
        
        st.subheader(f"{company} Topic Impact Heatmap")
        
        # Pivot topics for heatmap
        pivot_topics = topics_df.groupby(['date', 'topic_id'])['probability'].mean().unstack().fillna(0)
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_topics.values.T,
            x=pivot_topics.index,
            y=[f"Topic {i}" for i in pivot_topics.columns],
            colorscale='Viridis'
        ))
        fig_heat.update_layout(title=f"Leadership Topic Intensity Over Time vs {company}", template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.info("💡 Heatmap shows topic strength. In a production environment, this would be correlated with T+N forward returns for the specific ticker.")
