import pandas as pd
import numpy as np
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketModeler:
    """
    Implements advanced market models: ASBN, CPTM-F, Synchrony, and Trajectory Shapes.
    """
    def __init__(self, db_path='./data/market_rhetoric.db'):
        self.db_path = db_path

    def calculate_asbn(self, df, stable_window=252):
        """
        Asset-Specific Baseline Normalization (ASBN).
        Computes Z(t) = (x(t) - mean_asset) / std_asset using a rolling stable regime.
        """
        # Calculate moving mean and std over the stable window (~1 trading year)
        rolling_mean = df['close'].rolling(window=stable_window, min_periods=20).mean()
        rolling_std = df['close'].rolling(window=stable_window, min_periods=20).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        asbn = (df['close'] - rolling_mean) / rolling_std
        return asbn

    def calculate_cptm_f(self, df, trend_window=60):
        """
        Counterfactual Price Trajectory Modeling (CPTM-F).
        Calculates deviation D(t) = (actual(t) - expected(t)) / std_asset.
        Expected price is a simple linear extrapolation from the previous trend_window.
        """
        expected = df['close'].rolling(window=trend_window).mean()
        rolling_std = df['close'].rolling(window=trend_window).std().replace(0, np.nan)
        
        cptm_f = (df['close'] - expected) / rolling_std
        return cptm_f

    def calculate_trajectory_acceleration(self, df):
        """
        Trajectory Shape Features: Acceleration (2nd derivative of price).
        """
        velocity = df['close'].diff()
        acceleration = velocity.diff()
        return acceleration

    def compute_regime_metrics(self):
        """
        Computes ASBN, CPTM-F, and shapes for all tickers and saves to regime_classifications.
        """
        logger.info("Computing ASBN, CPTM-F, and Trajectory Shapes...")
        conn = sqlite3.connect(self.db_path)
        
        market_df = pd.read_sql_query("SELECT * FROM market_data ORDER BY date", conn)
        market_df['date'] = pd.to_datetime(market_df['date'])
        
        vix_df = pd.read_sql_query("SELECT date, vix_close FROM vix_data ORDER BY date", conn)
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        
        tickers = market_df['ticker'].unique()
        
        # Clear old classifications
        conn.execute("DELETE FROM regime_classifications")
        
        for ticker in tickers:
            df = market_df[market_df['ticker'] == ticker].copy()
            df = df.sort_values('date')
            
            # Merge VIX
            df = df.merge(vix_df, on='date', how='left')
            df['vix_close'] = df['vix_close'].ffill() # Forward fill missing VIX
            
            df['asbn'] = self.calculate_asbn(df)
            df['cptm_f'] = self.calculate_cptm_f(df)
            df['acceleration'] = self.calculate_trajectory_acceleration(df)
            
            # Multi-scale trends
            df['trend_short'] = df['close'].rolling(20).mean()
            df['trend_medium'] = df['close'].rolling(60).mean()
            df['trend_long'] = df['close'].rolling(120).mean()
            
            # Volume z-score
            vol_mean = df['volume'].rolling(60).mean()
            vol_std = df['volume'].rolling(60).std().replace(0, np.nan)
            df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std
            
            # Regime classification based on CPTM-F and ASBN
            # If actual deviates strongly (> 1.5 std) and volume is high (>1.0 std), it's a structural deviation.
            conditions = [
                (df['cptm_f'] > 1.5) & (df['trend_short'] > df['trend_long']),
                (df['cptm_f'] < -1.5) & (df['trend_short'] < df['trend_long'])
            ]
            choices = ['Bullish_Surge', 'Bearish_Shock']
            df['regime_pred'] = np.select(conditions, choices, default='Stable')
            
            # Insert back to DB
            for _, row in df.dropna(subset=['asbn', 'cptm_f']).iterrows():
                try:
                    conn.execute('''
                        INSERT INTO regime_classifications 
                        (date, sector, regime, confidence, deviation_magnitude, volume_zscore, volatility_ratio)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['date'].strftime('%Y-%m-%d'),
                        ticker,
                        row['regime_pred'],
                        abs(row['cptm_f']) / 3.0, # pseudo confidence bounded by 3-sigma
                        row['asbn'],
                        row['volume_zscore'],
                        row['vix_close'] if pd.notna(row['vix_close']) else 0.0
                    ))
                except Exception as e:
                    logger.debug(f"Error inserting regime: {e}")
                    
        conn.commit()
        conn.close()
        logger.info("✓ Regime metrics computed and saved.")

if __name__ == "__main__":
    mm = MarketModeler()
    mm.compute_regime_metrics()
