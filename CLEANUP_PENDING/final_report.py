"""
BaatSeBharat - Final Speech-Market Influence Report
Generates a deduplicated, clean report for MKB + Fed speeches
with actual historical stock prices (past vs present).
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys

DB_PATH     = "./data/market_rhetoric.db"
IMPACT_CSV  = "./data/processed/speech_market_impact_full.csv"
LABELS_JSON = "./data/processed/bertopic_topic_info.csv"
OUTPUT      = "./data/processed/final_influence_report.txt"

SECTOR_TOPIC_MAP = {
    "Banking":      ["bank", "finance", "credit", "loan", "rbi", "interest", "monetary", "reserve", "financial", "system"],
    "IT":           ["digital", "technology", "startup", "innovation", "software", "internet", "data", "ai", "artificial"],
    "Pharma":       ["health", "medicine", "covid", "hospital", "vaccine", "doctor", "pharma"],
    "Auto":         ["vehicle", "road", "transport", "highway", "infrastructure", "electric", "automobile"],
    "Energy":       ["energy", "solar", "power", "oil", "gas", "renewable", "electricity"],
    "Agriculture":  ["farmer", "agriculture", "crop", "village", "water", "kisan", "food", "rural"],
    "Broad Market": ["economy", "growth", "gdp", "employment", "trade", "export", "inflation", "friend", "countryman"],
}

SECTOR_TICKERS = {
    "Banking":     ["HDFCBANK.NS", "ICICIBANK.NS", "^NSEBANK"],
    "IT":          ["TCS.NS", "INFY.NS", "^CNXIT"],
    "Pharma":      ["SUNPHARMA.NS", "^CNXPHARMA"],
    "Auto":        ["MARUTI.NS", "^CNXAUTO"],
    "Energy":      ["RELIANCE.NS", "^CNXENERGY"],
    "Agriculture": ["^NSEI", "^BSESN"],
    "Broad Market":["^NSEI", "^BSESN", "^GSPC"],
}

def classify_topic(keywords):
    kw_str = " ".join(keywords).lower()
    best, best_score = "Broad Market", 0
    for sector, kws in SECTOR_TOPIC_MAP.items():
        score = sum(1 for kw in kws if kw in kw_str)
        if score > best_score:
            best_score, best = score, sector
    return best

def main():
    # Load impact data
    df = pd.read_csv(IMPACT_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "past_price", "return_5d"])

    # Load speech topics from DB
    conn = sqlite3.connect(DB_PATH)
    speeches_df = pd.read_sql_query(
        """SELECT s.id, s.date, s.source, s.title, s.speaker,
                  td.topic_id
           FROM speeches s
           LEFT JOIN topic_distributions td ON s.id = td.speech_id AND td.model_name='BERTopic'
           WHERE s.date IS NOT NULL AND s.date != ''
           ORDER BY s.date DESC""",
        conn
    )
    conn.close()

    speeches_df["date"] = pd.to_datetime(speeches_df["date"], format="mixed", errors="coerce")
    speeches_df = speeches_df.dropna(subset=["date"])
    # Deduplicate: keep one row per (date, source, title) combo
    speeches_df = speeches_df.drop_duplicates(subset=["date", "source", "title"])

    # Load bertopic info
    topic_kws = {}
    if os.path.exists(LABELS_JSON):
        ti = pd.read_csv(LABELS_JSON)
        for _, row in ti.iterrows():
            topic_id = row.get("Topic", -1)
            name = str(row.get("Name", ""))
            # Parse keywords from the name like "0_bank_reserve_financial_system"
            parts = name.split("_")[1:]
            topic_kws[int(topic_id)] = parts

    lines = []
    lines.append("=" * 100)
    lines.append("  BAATSE BHARAT — FINAL SPEECH-MARKET INFLUENCE REPORT")
    lines.append("  Method: BERTopic (SBERT embeddings) | Sectors: NSE/BSE/NYSE | Horizon: T+5d, T+10d")
    lines.append("=" * 100)

    # Determine meaningful sample: MKB speeches with real historical data
    # Pick 10 MKB speeches with confirmed non-zero returns, plus all Fed speeches
    mkb = speeches_df[speeches_df["source"] == "Mann Ki Baat"].sort_values("date", ascending=False).head(30)
    fed = speeches_df[speeches_df["source"] == "Fed"].sort_values("date", ascending=False)
    combined = pd.concat([fed, mkb]).sort_values("date", ascending=False)

    shown_event_keys = set()
    event_count = 0

    for _, sp in combined.iterrows():
        event_key = f"{sp['date'].date()}_{sp['source']}_{str(sp['title'])[:40]}"
        if event_key in shown_event_keys:
            continue
        shown_event_keys.add(event_key)

        # Get topic
        tid = int(sp["topic_id"]) if pd.notna(sp.get("topic_id")) else -1
        kws = topic_kws.get(tid, [])
        sector = classify_topic(kws)
        relevant_tickers = SECTOR_TICKERS.get(sector, SECTOR_TICKERS["Broad Market"])

        # Get impact rows for this speech on relevant tickers
        sp_impact = df[
            (df["speech_id"] == sp["id"]) &
            (df["ticker"].isin(relevant_tickers))
        ].drop_duplicates(subset=["ticker"])

        if sp_impact.empty:
            # Try by date + source (for speeches not in impact CSV by ID)
            sp_impact = df[
                (df["date"].dt.date == sp["date"].date()) &
                (df["source"] == sp["source"]) &
                (df["ticker"].isin(relevant_tickers))
            ].drop_duplicates(subset=["ticker"])

        if sp_impact.empty:
            continue

        speaker = str(sp.get("speaker", "Unknown"))
        title   = str(sp.get("title", "Untitled"))[:85]
        kw_str  = ", ".join(kws[:6]) if kws else "N/A"

        lines.append(f"\n{'-' * 100}")
        lines.append(f"  EVENT    : {sp['date'].strftime('%Y-%m-%d')}  |  {sp['source']}  |  {speaker}")
        lines.append(f"  TITLE    : {title}")
        lines.append(f"  TOPIC    : {sector} Sector  |  Keywords: [{kw_str}]")
        lines.append(f"{'-' * 100}")
        lines.append(f"  {'Ticker':<16} {'Sector':<14} {'Past Price':>12} {'Price T+5':>12} {'Price T+10':>12} {'5D Return':>11} {'10D Return':>11} {'Influence'}")
        lines.append(f"  {'─'*16} {'─'*14} {'─'*12} {'─'*12} {'─'*12} {'─'*11} {'─'*11} {'─'*10}")

        for _, row in sp_impact.sort_values("ticker").iterrows():
            past   = row["past_price"]
            p5     = row["price_t5"]  if pd.notna(row["price_t5"])  else None
            p10    = row["price_t10"] if pd.notna(row["price_t10"]) else None
            r5     = row["return_5d"]
            r10    = row["return_10d"] if pd.notna(row["return_10d"]) else None

            p5_str  = f"{p5:,.2f}"   if p5  else "N/A"
            p10_str = f"{p10:,.2f}"  if p10 else "N/A"
            r5_str  = f"{r5*100:+.2f}%"
            r10_str = f"{r10*100:+.2f}%" if r10 is not None else "N/A"
            inf_str = "POSITIVE (+)" if r5 > 0.02 else ("NEGATIVE (-)" if r5 < -0.02 else "NEUTRAL (~)")

            lines.append(
                f"  {row['ticker']:<16} {row['sector']:<14} {past:>12,.2f} {p5_str:>12} {p10_str:>12} {r5_str:>11} {r10_str:>11} {inf_str}"
            )
        event_count += 1

    lines.append(f"\n{'=' * 100}")
    lines.append(f"  BERTOPIC SUMMARY — Topics Discovered")
    lines.append(f"{'=' * 100}")
    if topic_kws:
        ti_df = pd.read_csv(LABELS_JSON) if os.path.exists(LABELS_JSON) else pd.DataFrame()
        if not ti_df.empty:
            for _, row in ti_df[ti_df["Topic"] != -1].iterrows():
                tid   = int(row["Topic"])
                count = int(row.get("Count", 0))
                kws   = topic_kws.get(tid, [])
                sec   = classify_topic(kws)
                lines.append(f"  Topic {tid:>2}  |  {count:>4} speeches  |  Sector: {sec:<14}  |  Keywords: [{', '.join(kws[:6])}]")
    else:
        lines.append("  No topic data available.")

    lines.append(f"\n{'=' * 100}")
    lines.append(f"  Total unique events shown: {event_count}")
    lines.append(f"  Market data coverage: {len(df)} impact records across {df['ticker'].nunique()} tickers")
    lines.append(f"  Report complete.")

    report = "\n".join(lines)
    # Safe UTF-8 output to terminal
    sys.stdout.buffer.write((report + "\n").encode("utf-8", errors="replace"))

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
