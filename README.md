# BaatSeBharat — Mann Ki Baat Analytics Dashboard

A research platform for analysing Prime Minister Narendra Modi's *Mann Ki Baat* radio addresses. It combines **LLM-powered topic labelling** (via Groq), FinBERT sentiment analysis, and an interactive Streamlit dashboard to explore how policy themes correlate with market movements.

---

## Features

- **LLM Topic Labelling** — Groq (Llama 3) reads each transcript and assigns structured policy topics from a fixed 14-label taxonomy. Fully incremental: only new episodes are labelled; existing ones are never re-processed.
- **Sentiment Analysis** — FinBERT classifies speech tone (positive / neutral / negative) at the episode level.
- **Topic Modeling (legacy)** — Hybrid BERTopic + LDA + NMF ensemble, still available for comparison.
- **Market Correlation** — Maps speech topics to NSE/BSE sectors and measures stock returns at T+1, T+5, T+10 days after each episode.
- **Interactive Dashboard** — Streamlit app with temporal trends, episode explorer, geo-map, and sector heatmaps.

---

## Project Structure

```
BaatSeBharat/
├── .env                          # API keys (never commit)
├── requirements.txt
├── dashboard.py                  # Main Streamlit app
├── App_v2.py                     # Alternate app entry point
│
├── scripts/
│   ├── llm_topic_pipeline.py     # ← LLM topic labelling (Groq) [NEW]
│   ├── bert_train_and_report_v2.py
│   ├── init_database.py
│   └── ...
│
├── src/
│   ├── models/
│   │   ├── topic_modeling.py     # HybridTopicModeler (LDA + NMF + BERTopic)
│   │   └── ...
│   └── ...
│
├── data/
│   ├── market_rhetoric.db        # SQLite — speeches + market data
│   ├── episode_topics.csv        # Per-episode topic labels (LLM output)
│   └── quarterly_topics.csv      # Quarterly topic summaries (auto-derived)
│
└── transcripts/
    └── mann_ki_baat/             # Raw .txt transcript files
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API keys

Open `.env` and fill in:

```env
GROQ_API_KEY=your_groq_api_key_here   # https://console.groq.com/keys
GROQ_MODEL=llama3-8b-8192              # or llama3-70b-8192 for higher quality

FRED_API_KEY=your_fred_key_here        # optional, for macro data
```

### 3. Run the dashboard

```bash
streamlit run dashboard.py
```

---

## LLM Topic Pipeline

The pipeline labels each episode with topics from this fixed taxonomy:

| Label | Label |
|---|---|
| Agriculture & Rural Development | Healthcare & Pandemic Response |
| Innovation & Technology | Education & Learning |
| Environment & Water Conservation | Infrastructure & Connectivity |
| Culture & Heritage | Yoga & Wellness |
| Defence & National Security | Women Empowerment & Social Justice |
| Economy & Financial Policy | Governance & Democracy |
| International Relations & Geopolitics | General / Mixed Theme |

### Usage

```bash
# Label only new episodes (standard workflow — token-efficient)
python scripts/llm_topic_pipeline.py

# Preview what would be labelled, without calling the API
python scripts/llm_topic_pipeline.py --dry-run

# Force re-label a specific episode
python scripts/llm_topic_pipeline.py --ep 126
```

**How it works:**
1. Reads `data/episode_topics.csv` to find already-labelled episodes.
2. Scans `transcripts/mann_ki_baat/` for any episodes not yet present.
3. Sends the first ~3,000 characters of each new transcript to Groq.
4. Parses the structured JSON response and appends to `episode_topics.csv`.
5. Rebuilds `quarterly_topics.csv` from the updated episode data.

Each run only calls the API for genuinely new episodes — no wasted tokens.

---

## Topic Categories

The LLM output per episode includes:

- `primary_topic` — the dominant policy theme
- `secondary_topic_1`, `secondary_topic_2` — supporting themes
- `confidence` — model confidence score (0–1)
- `key_theme_1/2/3` — short descriptive phrases

---

## Dependencies

| Package | Purpose |
|---|---|
| `groq` | LLM API client (topic labelling) |
| `streamlit` | Dashboard framework |
| `pandas` / `numpy` | Data processing |
| `bertopic` | Legacy topic modeling |
| `sentence-transformers` | SBERT embeddings |
| `transformers` | FinBERT sentiment |
| `plotly` | Interactive charts |
| `yfinance` | Market data |
| `python-dotenv` | `.env` loading |

---

## Authors

Disha Kataria

## Acknowledgments

- Data: *Mann Ki Baat* radio programme transcripts
- Built under the guidance of Prof. Jugal Manek
- LLM topic labelling powered by Groq (Llama 3)
- Sentiment analysis via ProsusAI/FinBERT