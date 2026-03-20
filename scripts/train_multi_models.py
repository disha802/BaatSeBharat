import sqlite3
import pandas as pd
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.models.topic_modeling import HybridTopicModeler

logger = setup_logger("MultiTopicTraining")
DB_PATH = './data/market_rhetoric.db'

def train_and_save_model(conn, name, query):
    logger.info(f"--- Training Topic Model: {name} ---")
    try:
        df_subset = pd.read_sql_query(query, conn)
        docs = df_subset['processed_text'].tolist()
        speech_ids = df_subset['id'].tolist()
        
        if len(docs) < 2:
            logger.warning(f"Not enough data for model '{name}' ({len(docs)} documents).")
            return
            
        n_topics = min(10, len(docs))
        modeler = HybridTopicModeler(n_topics=n_topics)
        embeddings = np.random.rand(len(docs), 384) # mock embeddings
        
        os.makedirs('./data/processed', exist_ok=True)
        consensus, dists = modeler.fit_ensemble(docs, embeddings)
        
        # Save to npy
        filename = f'topic_distributions_{name.lower().replace(" ", "_")}.npy'
        np.save(f'./data/processed/{filename}', consensus)
        logger.info(f"✓ Saved {filename}")
        
        # Persist to DB with retry/timeout logic
        cursor = conn.cursor()
        cursor.execute("DELETE FROM topic_distributions WHERE model_name = ?", (name,))
        for i, speech_id in enumerate(speech_ids):
            if i >= len(consensus): break
            for topic_id, prob in enumerate(consensus[i]):
                cursor.execute('''
                    INSERT OR REPLACE INTO topic_distributions 
                    (speech_id, topic_id, probability, model_name)
                    VALUES (?, ?, ?, ?)
                ''', (int(speech_id), topic_id, float(prob), name))
        conn.commit()
        logger.info(f"✓ Persisted {name} model to DB.")
    except Exception as e:
        logger.error(f"Failed training {name}: {e}")
        conn.rollback()

def run():
    # Use a larger timeout for SQLite to handle existing connections
    conn = sqlite3.connect(DB_PATH, timeout=30)
    
    model_tasks = [
        ("Combined", "SELECT id, processed_text FROM speeches WHERE processed_text IS NOT NULL AND processed_text != ''"),
        ("Fed", "SELECT id, processed_text FROM speeches WHERE source='Fed' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("ECB", "SELECT id, processed_text FROM speeches WHERE source='ECB' AND processed_text IS NOT NULL AND processed_text != ''"),
        ("Mann Ki Baat", "SELECT id, processed_text FROM speeches WHERE source IN ('Mann Ki Baat', 'MKB') AND processed_text IS NOT NULL AND processed_text != ''")
    ]
    
    for name, query in model_tasks:
        train_and_save_model(conn, name, query)
        
    conn.close()
    logger.info("=== Multi-Source Topic Training Complete ===")

if __name__ == "__main__":
    run()
