import sqlite3
import pandas as pd

DB_PATH = './data/market_rhetoric.db'

def check():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
        print("--- Speech Sources ---")
        print(df)
        
        df_m = pd.read_sql_query("SELECT ticker, COUNT(*) as count FROM market_data GROUP BY ticker", conn)
        print("\n--- Market Data ---")
        print(df_m)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check()
