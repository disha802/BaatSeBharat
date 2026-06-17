"""
BaatSeBharat – LLM Topic Labelling Pipeline (Groq)
────────────────────────────────────────────────────
Incremental pipeline that:
  1. Reads the existing episode_topics.csv to know which episodes are
     already labelled.
  2. Scans transcript files for NEW episodes (not yet in the CSV).
  3. Sends each new transcript to Groq with a structured prompt.
  4. Parses the LLM response and appends to episode_topics.csv.
  5. Re-derives quarterly_topics.csv from the updated episode data.

Usage:
  python scripts/llm_topic_pipeline.py            # label all new episodes
  python scripts/llm_topic_pipeline.py --dry-run  # preview without API calls
  python scripts/llm_topic_pipeline.py --ep 125   # re-label a specific episode
"""

import os
import re
import sys
import json
import glob
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

# Force UTF-8 output so Unicode prints fine on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from dotenv import load_dotenv

# ─── Load env ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ─── Paths ────────────────────────────────────────────────────────────────────
TRANSCRIPT_DIR   = ROOT / "transcripts" / "mann_ki_baat"
EPISODE_CSV      = ROOT / "data" / "episode_topics.csv"
QUARTERLY_CSV    = ROOT / "data" / "quarterly_topics.csv"

# ─── Topic taxonomy (same labels your existing data uses) ─────────────────────
TOPIC_LABELS = [
    "Agriculture & Rural Development",
    "Healthcare & Pandemic Response",
    "Innovation & Technology",
    "Education & Learning",
    "Environment & Water Conservation",
    "Infrastructure & Connectivity",
    "Culture & Heritage",
    "Yoga & Wellness",
    "Defence & National Security",
    "Women Empowerment & Social Justice",
    "Economy & Financial Policy",
    "Governance & Democracy",
    "International Relations & Geopolitics",
    "General / Mixed Theme",
]

# ─── Prompt template ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert political analyst specialising in Indian governance and
    public policy. Your task is to read excerpts from PM Modi's "Mann Ki Baat"
    radio programme transcripts and identify the top policy/thematic topics
    discussed.

    You must classify topics ONLY from this fixed list:
    {labels}

    Reply ONLY with a valid JSON object — no markdown, no explanation.
    Format:
    {{
      "primary_topic": "<label>",
      "secondary_topics": ["<label>", "<label>"],
      "confidence": <0.0-1.0>,
      "key_themes": ["short phrase 1", "short phrase 2", "short phrase 3"]
    }}
""").strip()

USER_PROMPT_TEMPLATE = textwrap.dedent("""
    Episode {episode_num} — {date}

    Transcript excerpt (first ~3000 characters):
    ---
    {excerpt}
    ---

    Identify the primary topic, up to 3 secondary topics, and 3 key short
    phrases that best summarise this episode.
""").strip()


# ─── Groq client (lazy import) ────────────────────────────────────────────────
def get_groq_client():
    try:
        from groq import Groq
    except ImportError:
        print("[ERROR] groq package not installed. Run: pip install groq")
        sys.exit(1)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("[ERROR] GROQ_API_KEY is not set. Edit your .env file and paste your key.")
        sys.exit(1)

    return Groq(api_key=GROQ_API_KEY)


# ─── Transcript helpers ───────────────────────────────────────────────────────
def parse_episode_number(filename: str) -> int | None:
    """Extract episode number from any transcript filename variant."""
    # Try: mann_ki_baat_20190630_Mann_Ki_Baat_-_Episode_1.txt
    m = re.search(r"Episode[_\s]+(\d+)", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Try: mann_ki_baat_1.txt
    m = re.match(r"mann_ki_baat_(\d+)\.txt$", filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def parse_episode_date(content: str) -> str | None:
    """Extract date from the first line of the transcript."""
    first_line = content.split("\n")[0].strip()
    m = re.search(r"\(([^)]+)\)", first_line)
    if m:
        date_str = m.group(1).strip()
        # Normalise: collapse multiple spaces, strip stray whitespace
        date_str = re.sub(r"\s+", " ", date_str).strip()
        for fmt in ["%d %b, %Y", "%d %B, %Y", "%d %b %Y", "%d %B %Y",
                    "%-d %b, %Y", "%-d %B, %Y", "%B %d, %Y"]:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue
        # Last resort: let dateutil parse it
        try:
            from dateutil import parser as dateutil_parser
            return dateutil_parser.parse(date_str, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def derive_quarter(date_str: str) -> str:
    """Convert YYYY-MM-DD → YYYYQn"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}Q{q}"
    except Exception:
        return "UnknownQ"


def load_transcript(path: Path) -> tuple[int | None, str | None, str]:
    """Return (episode_num, date_str, full_text)"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ep_num   = parse_episode_number(path.name)
    date_str = parse_episode_date(text)
    return ep_num, date_str, text


# ─── Discover new episodes ────────────────────────────────────────────────────
def get_already_labelled_episodes() -> set[int]:
    """Return set of episode numbers already present in episode_topics.csv."""
    if not EPISODE_CSV.exists():
        return set()
    try:
        df = pd.read_csv(EPISODE_CSV)
        if "episode" in df.columns:
            return set(df["episode"].dropna().astype(int).tolist())
    except Exception:
        pass
    return set()


def discover_new_transcripts(target_episode: int | None = None) -> list[dict]:
    """
    Returns a list of dicts with keys: path, episode, date, text
    Only includes episodes not yet in episode_topics.csv.
    If target_episode is given, only that episode is returned (for re-labelling).
    """
    already_done = get_already_labelled_episodes()

    # Use the canonical dated filenames (avoid duplicates with _None_ variants)
    pattern = str(TRANSCRIPT_DIR / "mann_ki_baat_2*.txt")
    files   = glob.glob(pattern)

    # Fall back to numbered files if dated ones are missing
    if not files:
        pattern = str(TRANSCRIPT_DIR / "mann_ki_baat_[0-9]*.txt")
        files   = glob.glob(pattern)

    seen_episodes: set[int] = set()
    new_episodes: list[dict] = []

    for fpath in sorted(files):
        ep_num, date_str, text = load_transcript(Path(fpath))
        if ep_num is None:
            continue
        if ep_num in seen_episodes:
            continue
        seen_episodes.add(ep_num)

        # Force re-label if --ep flag used
        if target_episode is not None:
            if ep_num == target_episode:
                new_episodes.append({"path": fpath, "episode": ep_num,
                                     "date": date_str, "text": text})
            continue

        if ep_num not in already_done:
            new_episodes.append({"path": fpath, "episode": ep_num,
                                 "date": date_str, "text": text})

    return new_episodes


# ─── LLM call ────────────────────────────────────────────────────────────────
def call_groq(client, episode_num: int, date_str: str | None, text: str) -> dict:
    """
    Send transcript excerpt to Groq and return parsed JSON result.
    Returns a dict with keys: primary_topic, secondary_topics, confidence, key_themes
    """
    labels_str = "\n".join(f"  - {l}" for l in TOPIC_LABELS)
    system_msg = SYSTEM_PROMPT.format(labels=labels_str)

    excerpt = text[:3000]  # ~750 tokens, well within context limits
    user_msg = USER_PROMPT_TEMPLATE.format(
        episode_num=episode_num,
        date=date_str or "Unknown",
        excerpt=excerpt,
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON block from text
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(m.group(0)) if m else {}

    # Normalise keys
    primary   = result.get("primary_topic", "General / Mixed Theme")
    secondary = result.get("secondary_topics", [])
    if not isinstance(secondary, list):
        secondary = []
    confidence = float(result.get("confidence", 0.8))
    key_themes = result.get("key_themes", [])
    if not isinstance(key_themes, list):
        key_themes = []

    # Guard: ensure labels are in our taxonomy
    if primary not in TOPIC_LABELS:
        primary = "General / Mixed Theme"
    secondary = [s for s in secondary if s in TOPIC_LABELS][:3]

    return {
        "primary_topic":    primary,
        "secondary_topics": secondary,
        "confidence":       round(confidence, 3),
        "key_themes":       key_themes[:3],
    }


# ─── CSV persistence ──────────────────────────────────────────────────────────
def load_or_create_episode_csv() -> pd.DataFrame:
    if EPISODE_CSV.exists():
        try:
            return pd.read_csv(EPISODE_CSV)
        except Exception:
            pass

    return pd.DataFrame(columns=[
        "episode", "date", "quarter",
        "primary_topic", "secondary_topic_1", "secondary_topic_2",
        "confidence", "key_theme_1", "key_theme_2", "key_theme_3",
        "labelled_by",
    ])


def append_episode_row(df: pd.DataFrame, ep: dict, result: dict) -> pd.DataFrame:
    """Append one episode's LLM result to the dataframe. Remove old row if re-labelling."""
    ep_num  = ep["episode"]
    date    = ep["date"] or ""
    quarter = derive_quarter(date) if date else "UnknownQ"
    sec     = result["secondary_topics"]
    themes  = result["key_themes"]

    # Drop any existing row for this episode (for re-labelling)
    df = df[df["episode"] != ep_num].copy()

    new_row = {
        "episode":           ep_num,
        "date":              date,
        "quarter":           quarter,
        "primary_topic":     result["primary_topic"],
        "secondary_topic_1": sec[0] if len(sec) > 0 else "",
        "secondary_topic_2": sec[1] if len(sec) > 1 else "",
        "confidence":        result["confidence"],
        "key_theme_1":       themes[0] if len(themes) > 0 else "",
        "key_theme_2":       themes[1] if len(themes) > 1 else "",
        "key_theme_3":       themes[2] if len(themes) > 2 else "",
        "labelled_by":       f"groq/{GROQ_MODEL}",
    }

    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def save_episode_csv(df: pd.DataFrame):
    df_sorted = df.sort_values("episode").reset_index(drop=True)
    df_sorted.to_csv(EPISODE_CSV, index=False)
    print(f"  [OK] Saved {len(df_sorted)} episodes -> {EPISODE_CSV}")


# ─── Regenerate quarterly summary ────────────────────────────────────────────
def rebuild_quarterly_csv(episode_df: pd.DataFrame):
    """
    Re-derive quarterly_topics.csv from the updated episode data.
    For each quarter, rank topics by frequency of appearance.
    """
    rows = []
    quarters = episode_df["quarter"].dropna().unique()

    for q in sorted(quarters):
        q_df = episode_df[episode_df["quarter"] == q]

        # Collect all topic mentions (primary + secondary) for this quarter
        topic_counts: dict[str, int] = {}
        for _, row in q_df.iterrows():
            for col in ["primary_topic", "secondary_topic_1", "secondary_topic_2"]:
                t = row.get(col, "")
                if t and t in TOPIC_LABELS:
                    topic_counts[t] = topic_counts.get(t, 0) + 1

        # Rank by frequency
        ranked = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        for rank, (topic, count) in enumerate(ranked[:5], start=1):
            # Gather key themes from episodes with this as primary topic
            ep_with_topic = q_df[q_df["primary_topic"] == topic]
            themes = []
            for _, row in ep_with_topic.iterrows():
                for tc in ["key_theme_1", "key_theme_2", "key_theme_3"]:
                    t = row.get(tc, "")
                    if t:
                        themes.append(t)
            top_words_str = ", ".join(themes[:8]) if themes else topic

            rows.append({
                "Quarter":    q,
                "Rank":       rank,
                "Topic_Label": topic,
                "Top_Words":   top_words_str,
                "Episode_Count": count,
            })

    if rows:
        qdf = pd.DataFrame(rows)
        qdf.to_csv(QUARTERLY_CSV, index=False)
        print(f"  [OK] Saved {len(qdf)} quarterly rows -> {QUARTERLY_CSV}")
    else:
        print("  [!] No quarterly data to save.")


# ─── Main pipeline ────────────────────────────────────────────────────────────
def run_pipeline(dry_run: bool = False, target_episode: int | None = None):
    print("\n" + "=" * 65)
    print("  BaatSeBharat — LLM Topic Pipeline (Groq)")
    print("=" * 65)

    # 1. Discover new episodes
    new_eps = discover_new_transcripts(target_episode=target_episode)

    if not new_eps:
        print("\n[OK] No new episodes to label. Everything is up to date!")
        return

    print(f"\n  Found {len(new_eps)} new episode(s) to label:")
    for ep in new_eps:
        print(f"    Episode {ep['episode']:>4} — {ep['date'] or 'date unknown'}")

    if dry_run:
        print("\n  [DRY RUN] No API calls made. Remove --dry-run to proceed.")
        return

    # 2. Load existing CSV
    episode_df = load_or_create_episode_csv()
    client     = get_groq_client()

    # 3. Process each new episode
    print(f"\n  Labelling with {GROQ_MODEL} ...")
    success = 0
    errors  = 0

    for i, ep in enumerate(new_eps, 1):
        ep_num = ep["episode"]
        print(f"\n  [{i}/{len(new_eps)}] Episode {ep_num} ({ep['date'] or '?'}) ", end="", flush=True)
        try:
            result = call_groq(client, ep_num, ep["date"], ep["text"])
            episode_df = append_episode_row(episode_df, ep, result)
            print(f"-> {result['primary_topic']}  (conf={result['confidence']})")
            success += 1
        except Exception as e:
            print(f"[FAIL] ERROR: {e}")
            errors += 1

    # 4. Save episode_topics.csv
    print()
    save_episode_csv(episode_df)

    # 5. Rebuild quarterly summary
    rebuild_quarterly_csv(episode_df)

    # 6. Summary
    print(f"\n{'=' * 65}")
    print(f"  Done. {success} labelled, {errors} error(s).")
    print(f"{'=' * 65}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM topic labelling pipeline for BaatSeBharat transcripts"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover new episodes and preview, but don't call the API"
    )
    parser.add_argument(
        "--ep", type=int, default=None, metavar="N",
        help="Force re-label a specific episode number (e.g. --ep 125)"
    )
    args = parser.parse_args()

    run_pipeline(dry_run=args.dry_run, target_episode=args.ep)
