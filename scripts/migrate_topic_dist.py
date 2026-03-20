import sqlite3
import os

DB_PATH = './data/market_rhetoric.db'

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if model_name column already exists
        cursor.execute("PRAGMA table_info(topic_distributions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'model_name' not in columns:
            print("Adding model_name column to topic_distributions...")
            # SQLite handles ADD COLUMN with DEFAULT easily
            cursor.execute("ALTER TABLE topic_distributions ADD COLUMN model_name TEXT DEFAULT 'combined'")
            
            # Update UNIQUE constraint is harder in SQLite (needs table recreation)
            # But let's see if we can just drop the old index/constraint if it's named, 
            # or just recreate the table.
            
            print("Recreating topic_distributions table to update UNIQUE constraint...")
            cursor.execute("ALTER TABLE topic_distributions RENAME TO topic_distributions_old")
            
            cursor.execute('''
                CREATE TABLE topic_distributions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speech_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    model_name TEXT DEFAULT 'combined',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (speech_id) REFERENCES speeches(id),
                    UNIQUE(speech_id, topic_id, model_name)
                )
            ''')
            
            cursor.execute('''
                INSERT INTO topic_distributions (id, speech_id, topic_id, probability, model_name, created_at)
                SELECT id, speech_id, topic_id, probability, model_name, created_at FROM topic_distributions_old
            ''')
            
            cursor.execute("DROP TABLE topic_distributions_old")
            print("✓ Migration successful.")
        else:
            print("model_name column already exists. Skipping migration.")
            
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.commit()
        conn.close()

if __name__ == "__main__":
    migrate()
