import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import asyncio

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.logger import setup_logger
from src.data.centralized_scraper import CentralizedSpeechScraper
from src.data.market_data_downloader import MarketDataDownloader
from src.features.text_preprocessing import TextPreprocessor
from src.models.topic_modeling import HybridTopicModeler
from src.utils.db_utils import get_db_connection

logger = setup_logger("Prototype_V1")

DB_PATH = './data/market_rhetoric.db'


def compute_speech_market_impact():
    """
    For each speech in the DB, compute 1-, 5- and 10-day forward returns
    for every market ticker. Saves results to speech_market_impact table.
    """
    logger.info("Computing speech-market impact...")
    conn = get_db_connection(DB_PATH)

    speeches_df = pd.read_sql_query(
        "SELECT id, date, source FROM speeches WHERE date IS NOT NULL", conn
    )
    market_df = pd.read_sql_query(
        "SELECT date, ticker, returns FROM market_data WHERE returns IS NOT NULL", conn
    )

    if speeches_df.empty or market_df.empty:
        logger.warning("Not enough data for impact computation.")
        conn.close()
        return

    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df = market_df.set_index('date').sort_index()

    # Clear old impact data
    conn.execute("DELETE FROM speech_market_impact")

    tickers = market_df['ticker'].unique()
    inserted = 0

    for _, row in speeches_df.iterrows():
        try:
            event_date = pd.to_datetime(row['date'])
        except Exception:
            continue

        for ticker in tickers:
            ticker_data = market_df[market_df['ticker'] == ticker]['returns']

            def forward_return(n_days):
                future = ticker_data[ticker_data.index > event_date]
                future = future.iloc[:n_days] if len(future) >= n_days else future
                if future.empty:
                    return None
                # Cumulative return: (1+r1)(1+r2)... - 1
                return float(np.prod(1 + future.values) - 1)

            r1 = forward_return(1)
            r5 = forward_return(5)
            r10 = forward_return(10)

            # Abnormal return: r5 minus the mean 5-day return of the ticker
            mean_5d = float(ticker_data.rolling(5).sum().mean()) if len(ticker_data) > 5 else None
            abnormal = (r5 - mean_5d) if (r5 is not None and mean_5d is not None) else None

            try:
                conn.execute('''
                    INSERT INTO speech_market_impact
                    (speech_id, ticker, event_date, return_t1, return_t5, return_t10, abnormal_return)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (int(row['id']), ticker, row['date'], r1, r5, r10, abnormal))
                inserted += 1
            except Exception as e:
                logger.error(f"Impact insert error: {e}")

    conn.commit()
    conn.close()
    logger.info(f"✓ Saved {inserted} speech-market impact records.")


def run_prototype():
    logger.info("=== Starting Patch V1.2 Prototype (Unified: MKB + ECB + Fed) ===")

    # 1. Centralized Data Ingestion — all 3 sources
    logger.info("Step 1: Ingesting multi-source speech data (MKB + ECB + Fed)...")
    scraper = CentralizedSpeechScraper(db_path=DB_PATH)
    scraper._ensure_db_exists() # Ensure all tables (including topic_distributions) exist

    try:
        asyncio.run(scraper.scrape_all(days_back=3650))
    except Exception as e:
        logger.error(f"Incomplete ingestion: {e}")

    # 2. Market Data
    logger.info("Step 2: Downloading market data...")
    try:
        downloader = MarketDataDownloader()
        market_df = downloader.download_all_data()
        if market_df is not None:
            downloader.save_to_database(market_df)
    except Exception as e:
        logger.error(f"Market data error: {e}")

    # 3. Preprocessing all speeches
    logger.info("Step 3: Preprocessing speeches...")
    preprocessor = TextPreprocessor()
    conn = get_db_connection(DB_PATH)

    try:
        conn.execute("ALTER TABLE speeches ADD COLUMN processed_text TEXT")
    except Exception:
        pass  # Column already exists

    df_speeches = pd.read_sql_query(
        "SELECT id, full_text FROM speeches WHERE full_text IS NOT NULL AND full_text != ''", conn
    )

    processed_count = 0
    for _, row in df_speeches.iterrows():
        try:
            processed = preprocessor.preprocess(row['full_text'])
            conn.execute(
                "UPDATE speeches SET processed_text = ? WHERE id = ?",
                (processed, row['id'])
            )
            processed_count += 1
        except Exception as e:
            logger.warning(f"Preprocess error id={row['id']}: {e}")

    conn.commit()
    logger.info(f"✓ Preprocessed {processed_count} speeches.")

    # 4. Multi-Source Topic Modeling
    logger.info("Step 4: Multi-Source Topic Modeling...")
    
    def train_and_save_model(name, query):
        logger.info(f"Training Topic Model: {name}...")
        df_subset = pd.read_sql_query(query, conn)
        docs = df_subset['processed_text'].tolist()
        speech_ids = df_subset['id'].tolist()
        
        if len(docs) < 2:
            logger.warning(f"Not enough data for model '{name}' ({len(docs)} documents).")
            return
            
        n_topics = min(10, len(docs))
        modeler = HybridTopicModeler(n_topics=n_topics)
        embeddings = np.random.rand(len(docs), 384) # mock embeddings
        
        try:
            os.makedirs('./data/processed', exist_ok=True)
            consensus, dists = modeler.fit_ensemble(docs, embeddings)
            
            # Save to npy
            filename = f'topic_distributions_{name.lower().replace(" ", "_")}.npy'
            np.save(f'./data/processed/{filename}', consensus)
            logger.info(f"✓ Saved {filename}")
            
            # Persist to DB
            conn.execute("DELETE FROM topic_distributions WHERE model_name = ?", (name,))
            for i, speech_id in enumerate(speech_ids):
                if i >= len(consensus): break
                for topic_id, prob in enumerate(consensus[i]):
                    conn.execute('''
                        INSERT OR REPLACE INTO topic_distributions 
                        (speech_id, topic_id, probability, model_name)
                        VALUES (?, ?, ?, ?)
                    ''', (int(speech_id), topic_id, float(prob), name))
            conn.commit()
            logger.info(f"✓ Persisted {name} model to DB.")
        except Exception as e:
            logger.error(f"Failed training {name}: {e}")

    # Define tasks
    model_tasks = [
        ("Combined", "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''"),
        ("Fed", "SELECT id, processed_text FROM speeches WHERE source='Fed' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("ECB", "SELECT id, processed_text FROM speeches WHERE source='ECB' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("Mann Ki Baat", "SELECT id, processed_text FROM speeches WHERE source='Mann Ki Baat' AND processed_text IS NOT NULL AND processed_text != ''")
    ]
    
    for name, query in model_tasks:
        train_and_save_model(name, query)

    conn.close()

    # 5. Compute Speech-Market Impact
    logger.info("Step 5: Computing speech-event market impact...")
    try:
        compute_speech_market_impact()
    except Exception as e:
        logger.error(f"Impact computation failed: {e}")

    logger.info("=== Prototype V1.2 Run Complete ===")
    return True


if __name__ == "__main__":
    run_prototype()
