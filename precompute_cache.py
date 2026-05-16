# precompute_cache.py
# Run ONCE to generate all small cached CSVs the dashboard reads.
# Re-run whenever the source data changes.
# ===========================================================

import pandas as pd
import os

DATA_DIR   = "content"
CACHE_DIR  = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

IMPACT_CSV = os.path.join(DATA_DIR, "speech_market_impact_full.csv")
TOPIC_DST  = os.path.join(DATA_DIR, "topic_dataset.csv")
TOPICS_CSV = os.path.join(DATA_DIR, "final_topics.csv")

REGIME_THRESH = 0.005   # >+0.5% = Bull, <-0.5% = Bear

print("Loading raw impact data...")
df = pd.read_csv(IMPACT_CSV, parse_dates=["date"])
df = df.dropna(subset=["return_5d", "date"])

# ──────────────────────────────────────────────────────────
# 1. Sector × Weekly aggregation
# ──────────────────────────────────────────────────────────
print("Computing weekly sector returns...")
df["week"] = df["date"].dt.to_period("W").astype(str)

weekly = (
    df.groupby(["sector", "week"])[["return_5d", "return_10d"]]
    .mean()
    .reset_index()
    .sort_values(["sector", "week"])
)
weekly.to_csv(os.path.join(CACHE_DIR, "sector_weekly.csv"), index=False)
print(f"  -> sector_weekly.csv  ({len(weekly)} rows)")

# ──────────────────────────────────────────────────────────
# 2. Sector × Quarterly aggregation
# ──────────────────────────────────────────────────────────
print("Computing quarterly sector returns...")
df["quarter"] = df["date"].dt.to_period("Q").astype(str)

quarterly = (
    df.groupby(["sector", "quarter"])[["return_5d", "return_10d"]]
    .mean()
    .reset_index()
    .sort_values(["sector", "quarter"])
)
quarterly.to_csv(os.path.join(CACHE_DIR, "sector_quarterly.csv"), index=False)
print(f"  -> sector_quarterly.csv  ({len(quarterly)} rows)")

# ──────────────────────────────────────────────────────────
# 3. Regime summary table (both granularities)
# ──────────────────────────────────────────────────────────
print("Computing regime summary tables...")

def build_regime_table(agg_df, period_col):
    rows = []
    for sec in agg_df["sector"].unique():
        sdf = agg_df[agg_df["sector"] == sec].copy()
        last_ret    = sdf["return_5d"].iloc[-1]
        last_period = sdf[period_col].iloc[-1]
        avg_ret     = sdf["return_5d"].mean()
        bull_pct    = (sdf["return_5d"] > REGIME_THRESH).mean() * 100
        bear_pct    = (sdf["return_5d"] < -REGIME_THRESH).mean() * 100

        if last_ret > REGIME_THRESH:
            regime = "Bull"
        elif last_ret < -REGIME_THRESH:
            regime = "Bear"
        else:
            regime = "Neutral"

        rows.append({
            "sector":       sec,
            "regime":       regime,
            "last_period":  last_period,
            "last_ret_5d":  round(last_ret * 100, 3),
            "avg_ret_5d":   round(avg_ret * 100, 3),
            "bull_pct":     round(bull_pct, 1),
            "bear_pct":     round(bear_pct, 1),
            "neutral_pct":  round(100 - bull_pct - bear_pct, 1),
        })
    return pd.DataFrame(rows)

regime_w = build_regime_table(weekly,    "week")
regime_q = build_regime_table(quarterly, "quarter")

regime_w.to_csv(os.path.join(CACHE_DIR, "regime_weekly.csv"),    index=False)
regime_q.to_csv(os.path.join(CACHE_DIR, "regime_quarterly.csv"), index=False)
print(f"  -> regime_weekly.csv / regime_quarterly.csv")

# ──────────────────────────────────────────────────────────
# 4. Sector overall averages (for topic-strength overlay)
# ──────────────────────────────────────────────────────────
print("Computing sector overall averages...")
sector_avg = (
    df.groupby("sector")[["return_5d", "return_10d"]]
    .mean()
    .reset_index()
)
sector_avg.to_csv(os.path.join(CACHE_DIR, "sector_avg.csv"), index=False)
print(f"  -> sector_avg.csv  ({len(sector_avg)} rows)")

# ──────────────────────────────────────────────────────────
# 5. Sentiment timeline (from topic_dataset)
# ──────────────────────────────────────────────────────────
print("Computing sentiment timeline...")
try:
    # topic_dataset can be huge - read only needed columns
    ds = pd.read_csv(
        TOPIC_DST,
        usecols=["date", "positive", "negative"],
        parse_dates=["date"],
    )
    ds = ds.dropna(subset=["date"])
    ds["sentiment"] = ds["positive"] - ds["negative"]
    timeline = (
        ds.groupby("date")["sentiment"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    timeline.to_csv(os.path.join(CACHE_DIR, "sentiment_timeline.csv"), index=False)
    print(f"  -> sentiment_timeline.csv  ({len(timeline)} rows)")
except Exception as e:
    print(f"  WARNING: Could not compute sentiment timeline: {e}")

# ──────────────────────────────────────────────────────────
# 6. Topic enrichment (short labels)
# ──────────────────────────────────────────────────────────
print("Enriching topic data...")
try:
    topics = pd.read_csv(TOPICS_CSV)
    topics["short_kw"] = (
        topics["keywords"]
        .str.split(",")
        .str[:4]
        .str.join(", ")
    )
    topics.to_csv(os.path.join(CACHE_DIR, "topics_enriched.csv"), index=False)
    print(f"  -> topics_enriched.csv  ({len(topics)} rows)")
except Exception as e:
    print(f"  WARNING: Could not enrich topics: {e}")

print("\nAll cache files written to:", CACHE_DIR)
print("Done!")
