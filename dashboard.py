# dashboard.py
# ==========================================================
# BAATSEBHARAT | LRDR-MRPRA DASHBOARD
# LDA + NMF + FinBERT VERSION  — cache-backed fast edition
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="BaatSeBharat | Analytics",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# PATHS
# ==========================================================

DATA_DIR  = "content"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Source files (only used if cache is missing)
TOPIC_DATASET = os.path.join(DATA_DIR, "topic_dataset.csv")
FINAL_TOPICS  = os.path.join(DATA_DIR, "final_topics.csv")
NMF_TOPICS    = os.path.join(DATA_DIR, "nmf_topics.csv")
REPORT_PATH   = os.path.join(DATA_DIR, "final_influence_report.txt")

# Pre-computed cache files
C_SECTOR_WEEKLY    = os.path.join(CACHE_DIR, "sector_weekly.csv")
C_SECTOR_QUARTERLY = os.path.join(CACHE_DIR, "sector_quarterly.csv")
C_REGIME_WEEKLY    = os.path.join(CACHE_DIR, "regime_weekly.csv")
C_REGIME_QUARTERLY = os.path.join(CACHE_DIR, "regime_quarterly.csv")
C_SECTOR_AVG       = os.path.join(CACHE_DIR, "sector_avg.csv")
C_SENTIMENT        = os.path.join(CACHE_DIR, "sentiment_timeline.csv")
C_TOPICS           = os.path.join(CACHE_DIR, "topics_enriched.csv")

# ==========================================================
# CACHE CHECK
# ==========================================================

cache_files = [
    C_SECTOR_WEEKLY, C_SECTOR_QUARTERLY,
    C_REGIME_WEEKLY, C_REGIME_QUARTERLY,
    C_SECTOR_AVG, C_TOPICS
]

if not all(os.path.exists(f) for f in cache_files):
    st.error(
        "⚠️ Cache files missing. Run once from your terminal:\n\n"
        "```\npython precompute_cache.py\n```"
    )
    st.stop()

# ==========================================================
# THEME
# ==========================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0f172a;
    color: #f1f5f9;
}
.hero-title {
    color: #38bdf8;
    font-weight: 700;
    font-size: 2.4rem;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
}
.stCard {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
}
section[data-testid="stSidebar"] {
    background-color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# DATA LOADING  (cached — tiny files, instant reads)
# ==========================================================

SECTOR_COLORS = {
    "Banking":      "#38bdf8",
    "IT":           "#a78bfa",
    "Pharma":       "#34d399",
    "Auto":         "#fb923c",
    "Energy":       "#fbbf24",
    "Broad Market": "#f472b6",
}
REGIME_THRESH = 0.005


@st.cache_data
def load_sector_weekly():
    return pd.read_csv(C_SECTOR_WEEKLY)


@st.cache_data
def load_sector_quarterly():
    return pd.read_csv(C_SECTOR_QUARTERLY)


@st.cache_data
def load_regime_weekly():
    return pd.read_csv(C_REGIME_WEEKLY)


@st.cache_data
def load_regime_quarterly():
    return pd.read_csv(C_REGIME_QUARTERLY)


@st.cache_data
def load_sector_avg():
    return pd.read_csv(C_SECTOR_AVG)


@st.cache_data
def load_sentiment_timeline():
    if os.path.exists(C_SENTIMENT):
        return pd.read_csv(C_SENTIMENT, parse_dates=["date"])
    return pd.DataFrame()


@st.cache_data
def load_topics():
    return pd.read_csv(C_TOPICS)


@st.cache_data
def load_nmf_topics():
    return pd.read_csv(NMF_TOPICS)


@st.cache_data
def load_report():
    if os.path.exists(REPORT_PATH):
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(REPORT_PATH, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
    return ""


# ==========================================================
# LOAD
# ==========================================================

df_weekly     = load_sector_weekly()
df_quarterly  = load_sector_quarterly()
df_regime_w   = load_regime_weekly()
df_regime_q   = load_regime_quarterly()
df_sector_avg = load_sector_avg()
df_sentiment  = load_sentiment_timeline()
df_topics     = load_topics()
df_nmf        = load_nmf_topics()
report_text   = load_report()

# ==========================================================
# HELPERS
# ==========================================================

def parse_report_sections(text):
    return [
        s.strip() for s in
        text.split("------------------------------------------------")
        if "EVENT" in s
    ]


def regime_label(ret):
    if ret > REGIME_THRESH * 100:
        return "🟢 BULL", "#34d399"
    elif ret < -REGIME_THRESH * 100:
        return "🔴 BEAR", "#f87171"
    return "⚪ NEUTRAL", "#94a3b8"


def build_sector_fig(sdf, sector, period_col, rcol_pct, period_label):
    """Build a per-sector return chart with regime bands from pre-computed data."""
    color = SECTOR_COLORS.get(sector, "#94a3b8")

    fig = go.Figure()

    # ── Regime bands (group contiguous) ───────────────────
    blocks, cur, start, bg = [], None, None, None
    for _, row in sdf.iterrows():
        r = row[rcol_pct]
        if r > REGIME_THRESH * 100:
            regime, c = "Bull", "rgba(52,211,153,0.13)"
        elif r < -REGIME_THRESH * 100:
            regime, c = "Bear", "rgba(248,113,113,0.13)"
        else:
            regime, c = "Neutral", "rgba(148,163,184,0.06)"

        if regime != cur:
            if cur is not None:
                blocks.append((cur, start, row[period_col], bg))
            cur, start, bg = regime, row[period_col], c

    if cur is not None:
        blocks.append((cur, start, sdf.iloc[-1][period_col], bg))

    for idx, (b_regime, b_start, b_end, b_color) in enumerate(blocks):
        fig.add_vrect(
            x0=b_start, x1=b_end,
            fillcolor=b_color, layer="below", line_width=0,
            annotation_text=b_regime if idx % max(1, len(blocks) // 4) == 0 else "",
            annotation_font_size=8, annotation_font_color="#94a3b8",
            annotation_position="top left"
        )

    # ── Return line ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=sdf[period_col], y=sdf[rcol_pct],
        mode="lines+markers", name=sector,
        line=dict(color=color, width=2), marker=dict(size=4),
        hovertemplate=(
            f"{period_label}: %{{x}}<br>"
            f"Return: %{{y:.3f}}%<extra>{sector}</extra>"
        )
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#475569", line_width=1)

    last_ret = sdf.iloc[-1][rcol_pct]
    rlabel, rcolor = regime_label(last_ret)

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                f"<b>{sector}</b>  "
                f"<span style='color:{rcolor};font-size:13px'>{rlabel}</span>"
            ),
            font=dict(size=15)
        ),
        height=320,
        xaxis=dict(title=period_label, tickangle=-45, nticks=10),
        yaxis=dict(title="Avg Return (%)", zeroline=True, zerolinecolor="#475569"),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        margin=dict(t=55, b=50, l=50, r=15),
        showlegend=False
    )
    return fig


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bullish.png", width=50)
    st.markdown("## BaatSeBharat")
    st.markdown("---")
    page = st.radio(
        "SELECT VIEW",
        ["📊 OVERVIEW", "📈 MARKET DYNAMICS", "🔎 SPEECH AUDIT", "🧠 TOPIC EXPLORER"]
    )
    st.markdown("---")
    st.caption("v2.1 | cache-backed")

# ==========================================================
# OVERVIEW
# ==========================================================

if page == "📊 OVERVIEW":

    st.markdown(
        '<h1 class="hero-title">Market Influence Analytics</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Leadership discourse vs market behaviour.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("LDA Topics", len(df_topics))
    with c2:
        st.metric("NMF Topics", len(df_nmf))
    with c3:
        if not df_sentiment.empty:
            avg_sent = round(df_sentiment["sentiment"].mean(), 3)
            st.metric("Avg Sentiment", avg_sent)

    st.markdown("### Sentiment Timeline")
    if not df_sentiment.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sentiment["date"], y=df_sentiment["sentiment"],
            mode="lines", name="Sentiment",
            line=dict(color="#38bdf8", width=1.5)
        ))
        fig.update_layout(
            template="plotly_dark", height=420,
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Analysis Feed")
    if report_text:
        for sec in parse_report_sections(report_text)[:5]:
            st.markdown(
                f'<div class="stCard"><pre>{sec}</pre></div>',
                unsafe_allow_html=True
            )

# ==========================================================
# MARKET DYNAMICS
# ==========================================================

elif page == "📈 MARKET DYNAMICS":

    st.markdown(
        '<h1 class="hero-title">Market Dynamics — Sector Intelligence</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-subtitle">Per-sector return timelines, regime shifts and topic influence.</p>',
        unsafe_allow_html=True
    )

    # ── Controls ──────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        granularity = st.radio(
            "Time Granularity", ["Quarterly", "Weekly"],
            horizontal=True, key="mkt_gran"
        )
    with col_ctrl2:
        return_horizon = st.radio(
            "Return Horizon", ["5-Day (1 Week)", "10-Day (2 Weeks)"],
            horizontal=True, key="mkt_ret"
        )

    use_5d       = "5-Day" in return_horizon
    rcol_raw     = "return_5d"  if use_5d else "return_10d"
    rcol_pct_col = rcol_raw + "_pct"  # we'll create this below

    if granularity == "Weekly":
        df_agg   = df_weekly.copy()
        df_reg   = df_regime_w.copy()
        period_col   = "week"
        period_label = "Week"
    else:
        df_agg   = df_quarterly.copy()
        df_reg   = df_regime_q.copy()
        period_col   = "quarter"
        period_label = "Quarter"

    # Convert to % for display
    df_agg[rcol_pct_col] = df_agg[rcol_raw] * 100

    SECTORS = list(df_agg["sector"].unique())

    st.markdown("---")
    st.markdown("### 📊 Per-Sector Return + Regime Shift")

    left_sectors  = SECTORS[:len(SECTORS) // 2 + len(SECTORS) % 2]
    right_sectors = SECTORS[len(SECTORS) // 2 + len(SECTORS) % 2:]

    col_left, col_right = st.columns(2)

    with col_left:
        for sec in left_sectors:
            sdf = df_agg[df_agg["sector"] == sec].sort_values(period_col)
            st.plotly_chart(
                build_sector_fig(sdf, sec, period_col, rcol_pct_col, period_label),
                use_container_width=True,
                key=f"left_{sec}_{granularity}_{return_horizon}"
            )

    with col_right:
        for sec in right_sectors:
            sdf = df_agg[df_agg["sector"] == sec].sort_values(period_col)
            st.plotly_chart(
                build_sector_fig(sdf, sec, period_col, rcol_pct_col, period_label),
                use_container_width=True,
                key=f"right_{sec}_{granularity}_{return_horizon}"
            )

    # ── Regime summary table ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Current Market Regime — All Sectors")

    regime_display = df_reg.copy()
    regime_display["Current Regime"] = regime_display["regime"].map({
        "Bull":    "🟢 Bull",
        "Bear":    "🔴 Bear",
        "Neutral": "⚪ Neutral",
    })
    regime_display = regime_display.rename(columns={
        "sector":      "Sector",
        "last_period": f"Latest {period_label}",
        "last_ret_5d": "Latest Return (%)",
        "avg_ret_5d":  "Avg Return (%)",
        "bull_pct":    "🟢 Bull %",
        "bear_pct":    "🔴 Bear %",
        "neutral_pct": "⚪ Neutral %",
    })[[
        "Sector", "Current Regime", f"Latest {period_label}",
        "Latest Return (%)", "Avg Return (%)",
        "🟢 Bull %", "🔴 Bear %", "⚪ Neutral %"
    ]]
    st.dataframe(regime_display, use_container_width=True, hide_index=True)

    # ── Topic strength vs sector overlay ──────────────────
    st.markdown("---")
    st.markdown("### 🧩 Topic Strength vs Sector Returns")

    sa = df_sector_avg.copy()
    rcol_avg = "return_5d" if use_5d else "return_10d"
    sa["avg_pct"] = sa[rcol_avg] * 100

    fig_ts = go.Figure()

    fig_ts.add_trace(go.Bar(
        x=df_topics["short_kw"],
        y=df_topics["score"],
        name="Topic Strength (LDA)",
        marker_color="#a78bfa",
        text=df_topics["score"].round(3),
        textposition="outside",
        hovertemplate=(
            "<b>Topic %{customdata[0]}</b><br>"
            "Keywords: %{customdata[1]}<br>"
            "Score: %{y:.3f}<extra></extra>"
        ),
        customdata=df_topics[["topic_id", "keywords"]].values,
        yaxis="y1"
    ))

    # Legend markers
    for _, row in sa.iterrows():
        fig_ts.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=SECTOR_COLORS.get(row["sector"], "#94a3b8"), size=10),
            name=f"{row['sector']} ({row['avg_pct']:.3f}%)",
            yaxis="y2"
        ))

    for _, row in sa.iterrows():
        fig_ts.add_hline(
            y=row["avg_pct"],
            line_color=SECTOR_COLORS.get(row["sector"], "#94a3b8"),
            line_dash="dash", line_width=1.5,
            annotation_text=row["sector"],
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color=SECTOR_COLORS.get(row["sector"], "#94a3b8"),
            yref="y2"
        )

    fig_ts.update_layout(
        template="plotly_dark", height=420,
        title="Topic Strength (bars) vs Sector Avg Returns (dashed lines)",
        xaxis=dict(title="Topic Keywords", tickangle=-30),
        yaxis=dict(title="Topic Score", side="left"),
        yaxis2=dict(title="Avg Return (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        margin=dict(t=55, b=130, l=55, r=90)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ==========================================================
# SPEECH AUDIT
# ==========================================================

elif page == "🔎 SPEECH AUDIT":

    st.markdown(
        '<h1 class="hero-title">Speech Intelligence Explorer</h1>',
        unsafe_allow_html=True
    )

    # Load full speech data only on this page
    @st.cache_data
    def load_speeches():
        df = pd.read_csv(
            TOPIC_DATASET,
            usecols=["date", "source", "filename", "text", "positive", "negative", "neutral"],
            parse_dates=["date"]
        )
        df = df.dropna(subset=["date"])
        df["sentiment"] = df["positive"] - df["negative"]
        df["title"] = df["filename"].str.replace(".txt", "", regex=False)
        return df

    df_speeches = load_speeches()

    search = st.text_input("Filter by keyword")
    filtered = df_speeches.copy()
    if search:
        filtered = filtered[
            filtered["title"].str.contains(search, case=False, na=False)
        ]

    st.dataframe(
        filtered[["date", "source", "title", "positive", "negative", "neutral"]]
        .sort_values("date", ascending=False),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    if not filtered.empty:
        selected_title = st.selectbox("Choose speech", filtered["title"].unique())
        speech = filtered[filtered["title"] == selected_title].iloc[0]

        st.markdown("### Speech Text")
        st.text_area("Transcript", speech["text"], height=280)

        fig_s = go.Figure(data=[go.Bar(
            x=["Positive", "Negative", "Neutral"],
            y=[speech["positive"], speech["negative"], speech["neutral"]],
            marker_color=["#34d399", "#f87171", "#94a3b8"]
        )])
        fig_s.update_layout(
            title="FinBERT Sentiment", template="plotly_dark",
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a"
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ==========================================================
# TOPIC EXPLORER
# ==========================================================

elif page == "🧠 TOPIC EXPLORER":

    st.markdown(
        '<h1 class="hero-title">Financial Topic Intelligence</h1>',
        unsafe_allow_html=True
    )

    st.markdown("### LDA Topics")
    st.dataframe(df_topics, use_container_width=True, hide_index=True)

    st.markdown("### NMF Topics")
    st.dataframe(df_nmf, use_container_width=True, hide_index=True)

    top_scores = (
        df_topics.sort_values("score", ascending=False).head(10).copy()
    )

    fig = go.Figure(go.Bar(
        x=top_scores["short_kw"],
        y=top_scores["score"],
        marker_color="#38bdf8",
        text=top_scores["score"].round(3),
        textposition="outside",
        customdata=top_scores[["topic_id", "keywords"]].values,
        hovertemplate=(
            "<b>Topic %{customdata[0]}</b><br>"
            "<b>Keywords:</b> %{customdata[1]}<br>"
            "Score: %{y:.3f}<extra></extra>"
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        title="Top Financial Topics by Score",
        xaxis=dict(title="Topic Keywords (first 4)", tickangle=-30),
        yaxis=dict(title="Topic Score"),
        height=420,
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        margin=dict(t=55, b=120, l=55, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)
