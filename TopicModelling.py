# modelling.py
import os
import re
import glob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# one-time downloads (only needed first run)
# nltk.download('stopwords')
# nltk.download('wordnet')

custom_stopwords = [
    "sir", "modi", "minister", "prime", "ji", "friends",
    "countrymen", "dear", "namaskar", "honourable", "respected"
]
stop_words = set(stopwords.words("english")) | set(custom_stopwords)
stop_words = list(stop_words)
lemmatizer = WordNetLemmatizer()

# --- text cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# --- load transcripts ---
def load_transcripts(path):
    data = []
    for fname in sorted(glob.glob(os.path.join(path, "mann_ki_baat_*.txt"))):
        with open(fname, "r", encoding="utf-8") as f:
            raw = f.read()

        # extract episode number + date if available
        ep_no, ep_date = None, None
        match = re.search(r"Episode\s+(\d+)\s+\((.*?)\)", raw)
        if match:
            ep_no = int(match.group(1))
            ep_date = match.group(2).strip()
        else:
            # fallback: derive episode number from filename
            base = os.path.basename(fname)
            ep_no = re.findall(r"\d+", base)
            ep_no = int(ep_no[0]) if ep_no else None

        cleaned = clean_text(raw)
        data.append({"episode": ep_no, "date": ep_date, "raw": raw, "clean": cleaned})
    return pd.DataFrame(data)

# --- main ---
if __name__ == "__main__":
    transcripts_path = os.path.join(os.path.dirname(__file__), "mann_ki_baat_transcripts")
    df = load_transcripts(transcripts_path)
    print(df.head())
    print(df.columns)
    print(len(df))


    # --- vectorize corpus ---
    vectorizer = CountVectorizer(
    max_df=0.60,      # ignore terms in >60% docs
    min_df=2,         # must appear in at least 2 docs
    max_features=5000,
    stop_words=stop_words,
    ngram_range=(1,2) # unigrams + bigrams
    )

    doc_term_matrix = vectorizer.fit_transform(df["clean"])

    # --- fit LDA model ---
    n_topics = 6   # you can tune this
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    terms = vectorizer.get_feature_names_out()

    # --- function to display topics ---
    def get_topics(model, feature_names, n_top_words=10):
        topics = []
        for idx, topic in enumerate(model.components_):
            top_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
            topics.append(top_terms)
        return topics

    topics = get_topics(lda, terms)

    # --- lightweight auto-labelling for readability ---
    def infer_label(top_words):
        if any(w in top_words for w in ["health", "doctor", "hospital", "patient", "disease", "covid", "oxygen", "vaccine"]):
            return "Healthcare & Pandemic Response"
        if any(w in top_words for w in ["yoga", "fitness", "wellness", "meditation", "spiritual"]):
            return "Yoga & Wellness"
        if any(w in top_words for w in ["water", "river", "lake", "cleanliness", "environment", "conservation", "sarovar"]):
            return "Environment & Water Conservation"
        if any(w in top_words for w in ["education", "student", "school", "learning", "mathematics", "teacher", "exam", "book"]):
            return "Education & Learning"
        if any(w in top_words for w in ["startup", "innovation", "technology", "science", "space", "research", "digital"]):
            return "Innovation & Technology"
        if any(w in top_words for w in ["woman", "selfhelp", "empowerment", "family", "children", "youth"]):
            return "Social Empowerment & Youth"
        if any(w in top_words for w in ["toy", "tribal", "culture", "heritage", "art", "craft", "festival", "tradition"]):
            return "Culture & Rural Development"
        if any(w in top_words for w in ["farmer", "crop", "agriculture", "soil", "village"]):
            return "Agriculture & Rural Economy"
        return "General / Mixed Theme"

    topic_labels = {i: infer_label(t) for i, t in enumerate(topics)}

    # --- compute topic distribution per doc ---
    topic_distributions = lda.transform(doc_term_matrix)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q")

    topic_cols = [f"Topic_{i}" for i in range(len(topics))]
    df_topics = pd.DataFrame(topic_distributions, columns=topic_cols)
    df = pd.concat([df, df_topics], axis=1)


    # --- summarize top 5 topics per quarter and save to CSV ---
    results = []

    for quarter, group in df.groupby("quarter"):
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

    # create DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "quarterly_topics.csv")
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n✅ Results saved to: {output_path}")
    print(results_df.head(10))


    # --- summarize top 5 topics per quarter ---
    print("\n==========  TOP 5 LABELED TOPICS PER QUARTER  ==========\n")
    for quarter, group in df.groupby("quarter"):
        mean_dist = group[topic_cols].mean().sort_values(ascending=False)
        top_topic_ids = mean_dist.index[:5]
        print(f"\n🗓️  Quarter: {quarter}")
        for tid in top_topic_ids:
            idx = int(tid.split("_")[1])
            label = topic_labels.get(idx, f"Topic {idx+1}")
            print(f"   • {label} → {', '.join(topics[idx][:8])}")
