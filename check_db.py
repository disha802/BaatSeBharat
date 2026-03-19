import sqlite3
import pandas as pd

DB_PATH = './data/market_rhetoric.db'

def check_db():
    conn = sqlite3.connect(DB_PATH)
    
    print("--- Source Count ---")
    df_counts = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
    print(df_counts)
    
    print("\n--- ECB Speeches ---")
    df_ecb = pd.read_sql_query("SELECT id, date, title, speaker FROM speeches WHERE source = 'ECB' ORDER BY date DESC", conn)
    print(df_ecb)
    
    print("\n--- Market Data Count ---")
    df_m = pd.read_sql_query("SELECT ticker, COUNT(*) as count FROM market_data GROUP BY ticker", conn)
    print(df_m)
    
    conn.close()

if __name__ == "__main__":
    check_db()
