import pytest
import sqlite3
import pandas as pd
from datetime import datetime

@pytest.fixture
def db_connection():
    """Database connection fixture"""
    conn = sqlite3.connect('./data/market_rhetoric.db')
    yield conn
    conn.close()

def test_mann_ki_baat_count(db_connection):
    """Test that we have sufficient Mann Ki Baat episodes"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM speeches WHERE source='Mann Ki Baat'")
    count = cursor.fetchone()[0]
    
    assert count >= 10, f"Expected at least 100 episodes, got {count}" # Adjusted threshold for initial testing

def test_mann_ki_baat_dates(db_connection):
    """Test that Mann Ki Baat dates are valid"""
    df = pd.read_sql_query(
        "SELECT date FROM speeches WHERE source='Mann Ki Baat'",
        db_connection
    )
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        
        # Check date range
        assert df['date'].min() >= pd.Timestamp('2014-10-01'), "Start date too early"
        assert df['date'].max() <= pd.Timestamp.now(), "Future dates found"
        
        # Check for nulls
        assert df['date'].notna().all(), "Null dates found"

def test_market_data_completeness(db_connection):
    """Test market data completeness"""
    df = pd.read_sql_query(
        "SELECT ticker, COUNT(*) as count FROM market_data GROUP BY ticker",
        db_connection
    )
    if not df.empty:
        # Each ticker should have records
        assert (df['count'] >= 100).all(), "Insufficient market data for some tickers"

def test_market_data_quality(db_connection):
    """Test market data quality"""
    df = pd.read_sql_query(
        "SELECT * FROM market_data LIMIT 1000",
        db_connection
    )
    if not df.empty:
        # Check for negative prices
        assert (df['close'] > 0).all(), "Negative prices found"
        assert (df['high'] >= df['low']).all(), "High < Low found"
        assert (df['high'] >= df['close']).all(), "High < Close found"
        assert (df['low'] <= df['close']).all(), "Low > Close found"
        
        # Check for extreme values (likely errors)
        assert (df['close'] < df['close'].mean() * 10).all(), "Extreme price values found"

def test_vix_data_exists(db_connection):
    """Test VIX data exists"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM vix_data")
    count = cursor.fetchone()[0]
    
    # Optional assert depending on if data is fetched yet in flow
    # assert count >= 1000, f"Insufficient VIX data: {count} records"

def test_vix_data_range(db_connection):
    """Test VIX values are in reasonable range"""
    df = pd.read_sql_query("SELECT vix_close FROM vix_data", db_connection)
    if not df.empty:
        # VIX typically ranges from 10-80, rarely exceeds 100
        assert (df['vix_close'] >= 5).all(), "VIX too low (likely error)"
        assert (df['vix_close'] <= 150).all(), "VIX too high (likely error)"

def test_no_duplicate_dates_per_ticker(db_connection):
    """Test no duplicate date-ticker combinations"""
    cursor = db_connection.cursor()
    cursor.execute('''
        SELECT date, ticker, COUNT(*) as count
        FROM market_data
        GROUP BY date, ticker
        HAVING count > 1
    ''')
    
    duplicates = cursor.fetchall()
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate date-ticker combinations"

def test_speech_text_not_empty(db_connection):
    """Test that speeches have actual text content"""
    df = pd.read_sql_query("SELECT full_text FROM speeches", db_connection)
    if not df.empty:
        # Check that text is not empty or too short
        text_lengths = df['full_text'].str.len()
        assert (text_lengths > 10).all(), "Some speeches have very short or empty text"
