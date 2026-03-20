import sqlite3
import pandas as pd

DB_PATH = './data/market_rhetoric.db'

def check():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Check schema
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"--- Tables ---")
        for table in tables:
            print(table[0])
        print("\n--- Topic Distributions Schema ---")
        cursor.execute("PRAGMA table_info(topic_distributions);")
        for info in cursor.fetchall():
            print(info)
            
        # Check counts
        df = pd.read_sql_query("SELECT source, COUNT(*) as count FROM speeches GROUP BY source", conn)
        print("\n--- Speech Sources ---")
        for index, row in df.iterrows():
            print(f"Source: {row['source']}, Count: {row['count']}")
        
        # Sample data
        df_sample = pd.read_sql_query("SELECT id, source, date, title FROM speeches LIMIT 10", conn)
        print("\n--- Sample Speeches ---")
        for index, row in df_sample.iterrows():
            print(f"ID: {row['id']}, Source: {row['source']}, Date: {row['date']}, Title: {row['title']}")
        
        df_topic = pd.read_sql_query("SELECT COUNT(*) as count FROM topic_distributions", conn)
        print(f"\n--- Topic Distributions Count: {df_topic['count'][0]} ---")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check()
