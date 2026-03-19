from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextEmbeddingGenerator:
    """
    Generate text embeddings using SBERT and IndicBERT
    """
    
    def __init__(self):
        logger.info("Loading embedding models...")
        
        # SBERT for English (384-dim)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # IndicBERT for Hindi (768-dim)
        self.indic_tokenizer = AutoTokenizer.frompretrained('ai4bharat/indic-bert')
        self.indic_model = AutoModel.from_pretrained('ai4bharat/indic-bert')
        self.indic_model.eval()
        
        logger.info("✓ Models loaded successfully")
    
    def embed_sbert(self, texts, batch_size=32, show_progress=True):
        """
        Generate SBERT embeddings
        Returns: numpy array of shape (n_texts, 384)
        """
        logger.info(f"Generating SBERT embeddings for {len(texts)} texts...")
        
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
        return embeddings
    
    def embed_indicbert(self, texts, batch_size=16, show_progress=True):
        """
        Generate IndicBERT embeddings for Hindi
        Returns: numpy array of shape (n_texts, 768)
        """
        logger.info(f"Generating IndicBERT embeddings for {len(texts)} texts...")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.indic_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.indic_model(**encoded)
                # Use CLS token
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"✓ Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def generate_and_save_embeddings(self, db_path='./data/market_rhetoric.db'):
        """
        Generate embeddings for all speeches and save
        """
        # Load speeches
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT id, processed_text, language FROM speeches WHERE processed_text IS NOT NULL",
            conn
        )
        
        logger.info(f"Generating embeddings for {len(df)} speeches...")
        
        # Separate by language
        english_df = df[df['language'].str.contains('English', case=False, na=False)]
        hindi_df = df[~df['language'].str.contains('English', case=False, na=False)]
        
        # Generate embeddings
        embeddings_dict = {}
        
        if len(english_df) > 0:
            logger.info(f"Processing {len(english_df)} English speeches...")
            english_embeddings = self.embed_sbert(english_df['processed_text'].tolist())
            
            for idx, speech_id in enumerate(english_df['id']):
                embeddings_dict[speech_id] = english_embeddings[idx]
        
        if len(hindi_df) > 0:
            logger.info(f"Processing {len(hindi_df)} Hindi speeches...")
            hindi_embeddings = self.embed_indicbert(hindi_df['processed_text'].tolist())
            
            for idx, speech_id in enumerate(hindi_df['id']):
                embeddings_dict[speech_id] = hindi_embeddings[idx]
        
        # Save embeddings
        logger.info("Saving embeddings...")
        
        # Save as numpy file
        os.makedirs('./data/processed', exist_ok=True)
        np.save('./data/processed/speech_embeddings.npy', embeddings_dict)
        
        # Also save mapping
        import pickle
        with open('./data/processed/speech_id_mapping.pkl', 'wb') as f:
            pickle.dump(list(embeddings_dict.keys()), f)
        
        conn.close()
        
        logger.info(f"✓ Saved embeddings for {len(embeddings_dict)} speeches")
        
        return embeddings_dict

if __name__ == "__main__":
    generator = TextEmbeddingGenerator()
    embeddings = generator.generate_and_save_embeddings()
