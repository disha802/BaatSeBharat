import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.nlp_engine import AdvancedNLPEngine
from src.analytics import AdvancedAnalytics
from datetime import datetime

st.set_page_config(
    page_title="Deep Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.markdown("<h1 style='text-align:center; color:#8e44ad;'>🧠 Financial & Rhetorical Intelligence</h1>", unsafe_allow_html=True)
st.markdown("### Advanced NLP, FinBERT Sentiment, and Statistical Causality")
st.markdown("---")

# Initialization
nlp = AdvancedNLPEngine()

# Sidebar
st.sidebar.header("🔬 Intelligence Controls")
analysis_mode = st.sidebar.radio("Analysis Depth", ["Standard", "Deep (BERTopic + FinBERT)"])

# Load processed data
if 'df' not in st.session_state:
    st.info("👈 Please load and process transcripts in the 'Topic Modeling' page first.")
    st.stop()

df = st.session_state.df

st.header("1️⃣ Contextual Topic Clusters (BERTopic)")

if analysis_mode == "Deep (BERTopic + FinBERT)":
    if st.button("🔥 Run Deep Neural Analysis"):
        with st.spinner("Executing Transformer-based Clustering (BERTopic)..."):
            # Sample for speed in demo if needed, but here we run full
            topics, probs, info = nlp.fit_bertopic(df['clean'].tolist())
            st.session_state.bertopic_info = info
            
            with st.spinner("Running FinBERT Sentiment Analysis..."):
                sentiments = nlp.analyze_sentiment(df['raw'].tolist())
                df['finbert_label'] = [s['label'] for s in sentiments]
                df['finbert_score'] = [s['score'] for s in sentiments]
                st.session_state.deep_df = df
        
        st.success("Analysis Complete!")

    if 'bertopic_info' in st.session_state:
        info = st.session_state.bertopic_info
        st.write("#### Discovered Contextual Themes")
        st.dataframe(info[['Topic', 'Count', 'Name', 'Representation']], use_container_width=True)
        
        # Visualization: Topic Probability Map
        # Note: In a real app, we'd use nlp.topic_model.visualize_topics() 
        # but for streamlit we use custom plotly to avoid iframe issues
        pass

else:
    st.warning("⚠️ Traditional NMF/LDA used. Switch to 'Deep' for Transformer-based analysis.")

# Section 2: FinBERT Sentiment Timeline
st.header("2️⃣ FinBERT Sentiment Intelligence")

if 'deep_df' in st.session_state:
    deep_df = st.session_state.deep_df
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sentiment vs Market Volatility")
        sent_color_map = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
        fig_sent = px.scatter(deep_df, x="date", y="finbert_score", color="finbert_label", 
                             size="finbert_score", title="FinBERT Sentiment Confidence Over Time",
                             color_discrete_map=sent_color_map)
        st.plotly_chart(fig_sent, use_container_width=True)
    
    with col2:
        st.subheader("Sentiment Mix")
        fig_pie = px.pie(deep_df, names="finbert_label", color="finbert_label",
                         color_discrete_map=sent_color_map)
        st.plotly_chart(fig_pie, use_container_width=True)
        
else:
    st.info("Sentiment data available after 'Deep Analysis' run.")

# Section 3: Causality (The "Prover")
st.header("3️⃣ Does Rhetoric *Lead* the Market?")

if 'market_df' in st.session_state and 'df' in st.session_state:
    topic_cols = st.session_state.topic_cols
    selected_topic = st.selectbox("Select Topic for Causality Testing:", topic_cols)
    
    # Merge and align
    m_df = st.session_state.market_data
    merged = pd.concat([df.set_index('date')[selected_topic], m_df], axis=1).ffill().dropna()
    
    if st.button("⚖️ Test Granger Causality"):
        p_vals = AdvancedAnalytics.test_granger_causality(merged, selected_topic, 'Price')
        
        # Plot p-values
        fig_p = px.bar(x=list(range(1, 11)), y=p_vals, title=f"Granger Causality P-Values (Lag 1-10): {selected_topic}",
                      labels={'x': 'Lag (Days)', 'y': 'P-Value'})
        fig_p.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Significance (5%)")
        st.plotly_chart(fig_p, use_container_width=True)
        
        if any(p < 0.05 for p in p_vals):
            st.success(f"**Found Significance!** Topics relating to {selected_topic} show statistical evidence of leading market movements.")
        else:
            st.warning("No significant causality found at this lag range.")

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>BaatSeBharat | Deep Rhetorical Intelligence Engine</div>", unsafe_allow_html=True)
