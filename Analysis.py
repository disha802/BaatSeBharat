import os
import re
import glob
import nltk
import spacy
import pandas as pd
import streamlit as st
import plotly.express as px
from nltk.corpus import stopwords
import plotly.graph_objects as go
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Download NLTK data if needed
try:
    stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Page configuration
st.set_page_config(
    page_title="Mann Ki Baat Topic Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.df = None
    st.session_state.topics = None
    st.session_state.topic_labels = None
    st.session_state.model = None
    st.session_state.vectorizer = None

# Enhanced stopwords
custom_stopwords = [
    "sir", "modi", "minister", "prime", "ji", "friends",
    "countrymen", "dear", "namaskar", "honourable", "respected",
    "tell", "say", "said", "come", "go", "make", "take", "give",
    "program", "programme", "today", "year", "time", "day",
    "india", "indian", "country", "people", "person", "thing",
    "way", "work", "effort", "initiative", "lot", "also",
    "become", "get", "start", "talk", "speak", "mention",
    "important", "great", "good", "new", "mann", "baat", "ki"
]
stop_words = set(stopwords.words("english")) | set(custom_stopwords)
stop_words = list(stop_words)

# Load spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

@st.cache_data
def clean_text_spacy(text):
    """Enhanced text cleaning with POS filtering"""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha
        and token.text not in stop_words
        and not token.is_stop
        and len(token.text) > 3  # Increased from 2 to 3
        and not token.like_num
        and token.pos_ in ['NOUN', 'VERB', 'ADJ']  # Only meaningful POS
    ]
    return " ".join(tokens)

# Load transcripts function
@st.cache_data
def load_transcripts(path):
    data = []
    for fname in sorted(glob.glob(os.path.join(path, "mann_ki_baat_*.txt"))):
        with open(fname, "r", encoding="utf-8") as f:
            raw = f.read()

        ep_no, ep_date = None, None
        match = re.search(r"Episode\s+(\d+)\s+\((.*?)\)", raw)
        if match:
            ep_no = int(match.group(1))
            ep_date = match.group(2).strip()
        else:
            base = os.path.basename(fname)
            ep_no = re.findall(r"\d+", base)
            ep_no = int(ep_no[0]) if ep_no else None

        cleaned = clean_text_spacy(raw)
        data.append({"episode": ep_no, "date": ep_date, "raw": raw, "clean": cleaned})
    return pd.DataFrame(data)

# Enhanced topic labeling function
def infer_label(top_words):
    if any(w in top_words for w in ["health", "doctor", "hospital", "patient", "disease", "covid", "oxygen", "vaccine", "medicine", "treatment"]):
        return "Healthcare & Pandemic Response"
    if any(w in top_words for w in ["yoga", "fitness", "wellness", "meditation", "spiritual", "exercise", "body"]):
        return "Yoga & Wellness"
    if any(w in top_words for w in ["water", "river", "lake", "cleanliness", "environment", "conservation", "sarovar", "pollution", "clean"]):
        return "Environment & Water Conservation"
    if any(w in top_words for w in ["education", "student", "school", "learning", "mathematics", "teacher", "exam", "book", "study", "knowledge"]):
        return "Education & Learning"
    if any(w in top_words for w in ["startup", "innovation", "technology", "science", "space", "research", "digital", "develop", "scientist"]):
        return "Innovation & Technology"
    if any(w in top_words for w in ["woman", "selfhelp", "empowerment", "family", "children", "youth", "daughter", "mother", "girl"]):
        return "Social Empowerment & Youth"
    if any(w in top_words for w in ["toy", "tribal", "culture", "heritage", "art", "craft", "festival", "tradition", "celebrate"]):
        return "Culture & Rural Development"
    if any(w in top_words for w in ["farmer", "crop", "agriculture", "soil", "village", "field", "harvest", "farm"]):
        return "Agriculture & Rural Economy"
    if any(w in top_words for w in ["soldier", "army", "defense", "security", "force", "border", "military", "nation"]):
        return "National Security & Defense"
    if any(w in top_words for w in ["sport", "athlete", "player", "game", "medal", "competition", "championship", "olympics"]):
        return "Sports & Athletics"
    return "General / Mixed Theme"

# Get topics with weight filtering
def get_topics(model, feature_names, n_top_words=10, min_weight=0.0005):
    topics = []
    for idx, topic in enumerate(model.components_):
        # Filter by minimum weight
        word_weights = [(feature_names[i], topic[i]) 
                       for i in range(len(feature_names)) 
                       if topic[i] > min_weight]
        word_weights.sort(key=lambda x: x[1], reverse=True)
        top_terms = [w[0] for w in word_weights[:n_top_words]]
        topics.append(top_terms)
    return topics

# Main UI
st.markdown('<div class="main-header">📊 Mann Ki Baat Topic Modeling Dashboard (Enhanced)</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    transcripts_path = st.text_input(
        "Transcripts Path",
        value="mann_ki_baat_transcripts",
        help="Path to the folder containing transcript files"
    )
    
    # Model selection
    model_type = st.selectbox(
        "Model Type",
        ["NMF (Recommended)", "LDA (Traditional)"],
        help="NMF often produces cleaner topics"
    )
    
    n_topics = st.slider("Number of Topics", min_value=3, max_value=15, value=8)
    max_df = st.slider("Max Document Frequency", min_value=0.3, max_value=0.9, value=0.5, step=0.05,
                      help="Remove words appearing in more than X% of documents")
    min_df = st.slider("Min Document Count", min_value=2, max_value=10, value=4,
                      help="Remove words appearing in fewer than X documents")
    max_features = st.slider("Max Vocabulary Size", min_value=500, max_value=2000, value=1000, step=100,
                            help="Limit total number of words considered")
    n_top_words = st.slider("Words per Topic", min_value=8, max_value=20, value=12)
    
    st.markdown("---")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Loading transcripts..."):
            try:
                df = load_transcripts(transcripts_path)
                
                if df.empty:
                    st.error("No transcripts found! Check the path.")
                else:
                    st.success(f"Loaded {len(df)} episodes")

                    # Vectorization
                    with st.spinner("Vectorizing text..."):
                        if model_type == "NMF (Recommended)":
                            # TF-IDF for NMF
                            vectorizer = TfidfVectorizer(
                                max_df=max_df,
                                min_df=min_df,
                                ngram_range=(1, 2),
                                stop_words=stop_words,
                                max_features=max_features,
                                sublinear_tf=True  # Use log scaling
                            )
                        else:
                            # Raw counts for LDA
                            vectorizer = CountVectorizer(
                                max_df=max_df,
                                min_df=min_df,
                                ngram_range=(1, 2),
                                stop_words=stop_words,
                                max_features=max_features
                            )
                        
                        doc_term_matrix = vectorizer.fit_transform(df["clean"])
                        st.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} words")

                    # Model training
                    with st.spinner(f"Training {model_type} model..."):
                        if model_type == "NMF (Recommended)":
                            model = NMF(
                                n_components=n_topics,
                                random_state=42,
                                max_iter=300,
                                alpha_W=0.0,  # No regularization (avoid zero issues)
                                alpha_H=0.0,
                                init='nndsvd',  # More stable initialization
                                solver='mu',  # Multiplicative update solver
                                beta_loss='frobenius'
                            )
                        else:
                            model = LatentDirichletAllocation(
                                n_components=n_topics,
                                learning_method='batch',
                                learning_decay=0.7,
                                max_iter=50,
                                evaluate_every=5,
                                random_state=42,
                                n_jobs=-1,
                                doc_topic_prior=0.1,  # Lower alpha = fewer topics per doc
                                topic_word_prior=0.01  # Lower beta = fewer words per topic
                            )
                        
                        model.fit(doc_term_matrix)

                    # Extract topics
                    terms = vectorizer.get_feature_names_out()
                    topics = get_topics(model, terms, n_top_words)
                    topic_labels = {i: infer_label(t) for i, t in enumerate(topics)}

                    # Topic distributions
                    topic_distributions = model.transform(doc_term_matrix)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["quarter"] = df["date"].dt.to_period("Q")

                    topic_cols = [f"Topic_{i}" for i in range(len(topics))]
                    df_topics = pd.DataFrame(topic_distributions, columns=topic_cols)
                    df = pd.concat([df, df_topics], axis=1)

                    # Topic coherence metric
                    with st.spinner("Calculating topic coherence..."):
                        texts = [d.split() for d in df["clean"]]
                        dictionary = Dictionary(texts)
                        corpus = [dictionary.doc2bow(t) for t in texts]

                        from gensim.models.ldamodel import LdaModel
                        lda_gensim = LdaModel(
                            corpus=corpus,
                            num_topics=n_topics,
                            id2word=dictionary,
                            passes=3,
                            random_state=42
                        )
                        coherence_model = CoherenceModel(
                            model=lda_gensim,
                            texts=texts,
                            dictionary=dictionary,
                            coherence='c_v'
                        )
                        coherence_score = coherence_model.get_coherence()
                        st.metric("Topic Coherence (c_v)", f"{coherence_score:.3f}", 
                                help="Higher is better. Good scores: >0.5")

                    # Store results in session state
                    st.session_state.model_trained = True
                    st.session_state.df = df
                    st.session_state.topics = topics
                    st.session_state.topic_labels = topic_labels
                    st.session_state.model = model
                    st.session_state.vectorizer = vectorizer
                    st.session_state.topic_cols = topic_cols
                    st.session_state.model_type = model_type

                    st.success("✅ Model trained successfully!")
                    st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# Main content
if not st.session_state.model_trained:
    st.info("👈 Configure parameters and click 'Train Model' to start the analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Default Topics", "8")
    with col2:
        st.metric("Max DF", "0.50")
    with col3:
        st.metric("Min DF", "4")
    with col4:
        st.metric("Max Features", "1000")
    
    st.markdown("### 🎯 Improvements in this version:")
    st.markdown("""
    - **Enhanced text preprocessing**: Filters by part-of-speech (nouns, verbs, adjectives only)
    - **Expanded stopwords**: Removes generic words that don't carry topical meaning
    - **Better vectorization**: Lower max_df and higher min_df to filter noise
    - **NMF option**: Often produces cleaner, more interpretable topics than LDA
    - **Stricter token filtering**: Minimum 3 characters (instead of 2)
    - **Weight-based topic extraction**: Only shows words with significant weights
    - **Vocabulary limiting**: Focuses on most important words
    """)
    
else:
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Topic Overview",
        "📊 Temporal Analysis",
        "🔍 Episode Explorer",
        "📥 Export Data"
    ])

    df = st.session_state.df
    topics = st.session_state.topics
    topic_labels = st.session_state.topic_labels
    topic_cols = st.session_state.topic_cols
    model_type = st.session_state.model_type
    
    # Tab 1: Topic Overview
    with tab1:
        st.header(f"Discovered Topics ({model_type})")
        
        for i, topic_words in enumerate(topics):
            label = topic_labels.get(i, f"Topic {i+1}")
            with st.expander(f"**Topic {i+1}: {label}**", expanded=(i < 3)):
                st.write(f"**Top Words:** {', '.join(topic_words[:12])}")
                
                # Word importance chart
                word_weights = st.session_state.model.components_[i]
                top_indices = word_weights.argsort()[:-13:-1]
                terms = st.session_state.vectorizer.get_feature_names_out()
                top_words = [terms[idx] for idx in top_indices]
                top_weights = [word_weights[idx] for idx in top_indices]
                
                fig = px.bar(
                    x=top_weights[::-1],
                    y=top_words[::-1],
                    orientation='h',
                    labels={'x': 'Weight', 'y': 'Word'},
                    title=f"Word Importance in {label}"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Temporal Analysis
    with tab2:
        st.header("Topic Distribution Over Time")
        
        # Quarter-wise analysis
        quarterly_data = []
        for quarter, group in df.groupby("quarter"):
            if pd.isna(quarter):
                continue
            mean_dist = group[topic_cols].mean()
            for i, col in enumerate(topic_cols):
                label = topic_labels.get(i, f"Topic {i+1}")
                quarterly_data.append({
                    "Quarter": str(quarter),
                    "Topic": label,
                    "Distribution": mean_dist[col]
                })
        
        quarterly_df = pd.DataFrame(quarterly_data)
        
        if not quarterly_df.empty:
            pivot_df = quarterly_df.pivot_table(
                index="Quarter", 
                columns="Topic", 
                values="Distribution",
                aggfunc='mean'
            )
            
            # Stacked area chart
            fig = go.Figure()
            for topic in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[topic],
                    name=topic,
                    mode='lines',
                    stackgroup='one',
                    fillcolor=None
                ))
            
            fig.update_layout(
                title="Topic Evolution Across Quarters",
                xaxis_title="Quarter",
                yaxis_title="Topic Distribution",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top topics per quarter
        st.subheader("Top 5 Topics Per Quarter")
        
        results = []
        for quarter, group in df.groupby("quarter"):
            if pd.isna(quarter):
                continue
            mean_dist = group[topic_cols].mean().sort_values(ascending=False)
            top_topic_ids = mean_dist.index[:5]
            
            for rank, tid in enumerate(top_topic_ids, start=1):
                idx = int(tid.split("_")[1])
                label = topic_labels.get(idx, f"Topic {idx+1}")
                top_words = ", ".join(topics[idx][:8])
                results.append({
                    "Quarter": str(quarter),
                    "Rank": rank,
                    "Topic": label,
                    "Top Words": top_words,
                    "Score": mean_dist[tid]
                })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            quarters = sorted(results_df["Quarter"].unique(), reverse=True)
            selected_quarter = st.selectbox("Select Quarter", quarters)
            quarter_data = results_df[results_df["Quarter"] == selected_quarter]
            
            st.dataframe(
                quarter_data[["Rank", "Topic", "Top Words", "Score"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No quarterly data available")
    
    # Tab 3: Episode Explorer
    with tab3:
        st.header("Explore Individual Episodes")
        
        episode_numbers = sorted(df["episode"].dropna().astype(int).unique(), reverse=True)
        if len(episode_numbers) > 0:
            selected_episode = st.selectbox("Select Episode", episode_numbers)
            
            episode_data = df[df["episode"] == selected_episode].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Episode", int(episode_data["episode"]))
                if pd.notna(episode_data["date"]):
                    st.metric("Date", episode_data["date"].strftime("%B %d, %Y"))
                else:
                    st.metric("Date", "N/A")
                st.metric("Quarter", str(episode_data["quarter"]))
            
            with col2:
                # Topic distribution for this episode
                topic_dist = {topic_labels.get(i, f"Topic {i+1}"): episode_data[f"Topic_{i}"] 
                              for i in range(len(topics))}
                topic_dist_sorted = dict(sorted(topic_dist.items(), key=lambda x: x[1], reverse=True))
                
                fig = px.bar(
                    x=list(topic_dist_sorted.values()),
                    y=list(topic_dist_sorted.keys()),
                    orientation='h',
                    title=f"Topic Distribution - Episode {int(selected_episode)}",
                    labels={'x': 'Probability', 'y': 'Topic'}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Show transcript excerpt
            with st.expander("📄 View Transcript Excerpt"):
                raw_text = episode_data["raw"]
                st.text_area("Raw Transcript", raw_text[:2000] + "..." if len(raw_text) > 2000 else raw_text, height=300)
        else:
            st.warning("No episodes found")
    
    # Tab 4: Export Data
    with tab4:
        st.header("Export Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export quarterly topics
            st.subheader("📊 Quarterly Topics")
            results = []
            for quarter, group in df.groupby("quarter"):
                if pd.isna(quarter):
                    continue
                mean_dist = group[topic_cols].mean().sort_values(ascending=False)
                top_topic_ids = mean_dist.index[:5]
                
                for rank, tid in enumerate(top_topic_ids, start=1):
                    idx = int(tid.split("_")[1])
                    label = topic_labels.get(idx, f"Topic {idx+1}")
                    top_words = ", ".join(topics[idx][:8])
                    results.append({
                        "Quarter": str(quarter),
                        "Rank": rank,
                        "Topic_Label": label,
                        "Top_Words": top_words
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                csv1 = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Quarterly Topics CSV",
                    data=csv1,
                    file_name="quarterly_topics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to export")
        
        with col2:
            # Export episode-level data
            st.subheader("📑 Episode-Level Data")
            export_cols = ["episode", "date", "quarter"] + topic_cols
            export_df = df[export_cols].copy()
            
            # Rename columns with labels
            rename_dict = {f"Topic_{i}": topic_labels.get(i, f"Topic {i+1}") for i in range(len(topics))}
            export_df = export_df.rename(columns=rename_dict)
            
            csv2 = export_df.to_csv(index=False)
            st.download_button(
                label="Download Episode Data CSV",
                data=csv2,
                file_name="episode_topics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        st.info("💡 Tip: Use these CSV files for further analysis in Excel, Python, or other tools.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit | Enhanced Topic Modeling with NMF/LDA</div>",
    unsafe_allow_html=True
)