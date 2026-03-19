import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.logger import setup_logger
from src.data.mann_ki_baat_scraper import MannKiBaatScraper
from src.data.market_data_downloader import MarketDataDownloader
from src.features.text_preprocessing import TextPreprocessor
from src.models.topic_modeling import HybridTopicModeler
import numpy as np

logger = setup_logger("Prototype_V1")

def run_prototype():
    logger.info("=== Starting Patch V1.0 Prototype ===")
    
    # 1. Minimal Data Ingestion
    logger.info("Step 1: Ingesting minimal data...")
    scraper = MannKiBaatScraper()
    # Mocking a list to avoid full scrape for prototype speed
    episodes = [
        {'url': 'dummy_url_1', 'title': 'Episode 100', 'date': datetime(2023, 1, 29).date(), 'full_text': 'India is progressing well in technology and self-reliance.'},
        {'url': 'dummy_url_2', 'title': 'Episode 101', 'date': datetime(2023, 2, 26).date(), 'full_text': 'Our farmers are the backbone of the economy. Budget 2023 focused on them.'}
    ]
    scraper.episodes = episodes
    scraper.save_to_database()

    # 2. Market Data
    downloader = MarketDataDownloader()
    # Just one ticker for prototype
    ticker_info = {'symbol': '^NSEI'}
    market_df = downloader.download_ticker_data(ticker_info, 'Broad Market', start_date='2023-01-01')
    if market_df is not None:
        downloader.save_to_database(market_df)

    # 3. Preprocessing
    logger.info("Step 2: Preprocessing...")
    preprocessor = TextPreprocessor()
    conn = sqlite3.connect('./data/market_rhetoric.db')
    
    # Ensure column exists
    try:
        conn.execute("ALTER TABLE speeches ADD COLUMN processed_text TEXT")
    except:
        pass
        
    df_speeches = pd.read_sql_query("SELECT id, full_text FROM speeches", conn)
    
    for idx, row in df_speeches.iterrows():
        processed = preprocessor.preprocess(row['full_text'])
        conn.execute("UPDATE speeches SET processed_text = ? WHERE id = ?", (processed, row['id']))
    conn.commit()

    # 4. Simple Topic Modeling (Mock/Small)
    logger.info("Step 3: Topic Modeling...")
    # Since we have only 2 speeches, we can't do full BERTopic easily here, 
    # but we'll run the logic to ensure the pipeline doesn't break.
    docs = [preprocessor.preprocess(e['full_text']) for e in episodes]
    modeler = HybridTopicModeler(n_topics=2)
    # Mocking embeddings for 2 docs
    embeddings = np.random.rand(len(docs), 384) 
    try:
        consensus, dists = modeler.fit_ensemble(docs, embeddings)
        np.save('./data/processed/topic_distributions_prototype.npy', consensus)
        logger.info("✓ Topic distributions saved.")
    except Exception as e:
        logger.warning(f"Topic modeling skipped (need more data for robust fit): {e}")

    logger.info("=== Prototype Run Complete ===")
    return True

if __name__ == "__main__":
    run_prototype()
