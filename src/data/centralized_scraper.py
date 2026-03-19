import sqlite3
import math
import pandas as pd
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import re
import os
import sys
import asyncio
from playwright.async_api import async_playwright

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class CentralizedSpeechScraper:
    """Centralized scraper for various leadership speeches (ECB, Fed, etc.)"""
    
    def __init__(self, db_path='./data/market_rhetoric.db'):
        self.db_path = db_path
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Ensure the database and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Speeches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speeches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                source TEXT,
                country TEXT,
                speaker TEXT,
                title TEXT,
                full_text TEXT,
                url TEXT,
                processed_text TEXT,
                UNIQUE(date, source, speaker, title)
            )
        ''')
        
        # Market data table (re-ensuring based on market_data_downloader.py)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT,
                ticker TEXT,
                sector TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                returns REAL,
                volatility REAL,
                PRIMARY KEY (date, ticker)
            )
        ''')
        
        # Table: Speech-Market Impact
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speech_market_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speech_id INTEGER,
                ticker TEXT,
                event_date TEXT,
                return_t1 REAL,
                return_t5 REAL,
                return_t10 REAL,
                abnormal_return REAL,
                FOREIGN KEY (speech_id) REFERENCES speeches(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_speeches(self, speeches):
        """Save a list of speech dictionaries to the database"""
        if not speeches:
            logger.warning("No speeches to save.")
            return 0
            
        conn = sqlite3.connect(self.db_path)
        saved_count = 0
        
        for speech in speeches:
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO speeches 
                    (date, source, country, speaker, title, full_text, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    speech.get('date'),
                    speech.get('source'),
                    speech.get('country'),
                    speech.get('speaker'),
                    speech.get('title'),
                    speech.get('full_text'),
                    speech.get('url')
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving speech: {e}")
                
        conn.commit()
        conn.close()
        logger.info(f"Saved {saved_count} speeches to database.")
        return saved_count

    # --- ECB Logic (Adapted from ecb.py) ---
    
    async def scrape_ecb(self, days_back=730):
        """Scrape ECB speeches"""
        logger.info(f"Scraping ECB speeches from last {days_back} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        speech_urls = set()
        url = "https://www.ecb.europa.eu/press/key/html/index.en.html"
        
        speeches = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="load")
            await asyncio.sleep(5)
            try:
                accept_button = await page.query_selector("button:has-text('Accept')")
                if accept_button: await accept_button.click()
            except: pass
            
            # Scroll to load content (infinite scroll)
            logger.info("Scrolling to load ECB speeches...")
            for i in range(10):  # More scrolls
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(3)
            
            links = await page.evaluate("Array.from(document.querySelectorAll('a')).map(a => a.href)")
            logger.info(f"Discovered {len(links)} links on ECB page.")
            
            for href in links:
                if href and '/press/key/date/' in href and 'sp' in href:
                    # New pattern: ecb.sp260309~6cfdbd02b7.en.html or old pattern: sp260309.en.html
                    # Relaxed check for ends with .en.html as it might have parameters
                    date_match = re.search(r'sp(\d{2})(\d{2})(\d{2})', href)
                    if date_match:
                        year, month, day = date_match.groups()
                        pub_date = datetime(2000 + int(year), int(month), int(day))
                        if pub_date >= cutoff_date:
                            speech_urls.add(href)
            
            logger.info(f"Filtered {len(speech_urls)} ECB speech URLs.")
            await browser.close()

        for url in speech_urls:
            content = self._scrape_ecb_content(url)
            if content:
                content['source'] = 'ECB'
                content['country'] = 'Europe'
                speeches.append(content)
            time.sleep(1)
            
        return self.save_speeches(speeches)

    def _scrape_ecb_content(self, url):
        """Scrapes individual ECB speech content"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            title = soup.find('h1', class_='title').get_text(strip=True) if soup.find('h1', class_='title') else 'N/A'
            date_tag = soup.find('p', class_='date')
            date_str = date_tag.get_text(strip=True) if date_tag else 'N/A'
            
            # Convert date_str to YYYY-MM-DD
            try:
                parsed_date = datetime.strptime(date_str, '%d %B %Y').strftime('%Y-%m-%d')
            except:
                parsed_date = date_str

            speaker = 'N/A'
            subtitle_tag = soup.find('p', class_='subtitle')
            if subtitle_tag:
                speaker_text = subtitle_tag.get_text(strip=True)
                if 'by' in speaker_text:
                    speaker = speaker_text.split('by')[-1].split(',')[0].strip()

            content_div = soup.find('div', class_='ecb-pressContent') or soup.find('main')
            full_text = ""
            if content_div:
                for p in content_div.find_all('p'):
                    if not p.find_parent('div', class_='contact'):
                        full_text += p.get_text(strip=True) + "\n\n"

            return {
                'url': url,
                'title': title,
                'date': parsed_date,
                'speaker': speaker,
                'full_text': full_text.strip()
            }
        except Exception as e:
            logger.error(f"Error scraping ECB {url}: {e}")
            return None

    # --- Fed Logic (Simplified from us_federalreserve.py - using requests instead of Selenium for speed if possible) ---
    
    async def scrape_fed(self, days_back=730):
        """Scrape US Federal Reserve speeches using Playwright for dynamic content"""
        logger.info(f"Scraping Fed speeches from last {days_back} days...")
        
        base_url = "https://www.federalreserve.gov"
        url = f"{base_url}/newsevents/speeches.htm"
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        speeches = []
        speech_links = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="load")
            await asyncio.sleep(5)
            try:
                await page.wait_for_selector(".itemTitle", timeout=10000)
            except:
                logger.warning("Timeout waiting for Fed speech list.")
            
            # Extract links and metadata
            items = await page.query_selector_all(".row")
            for item in items:
                try:
                    date_tag = await item.query_selector("time.itemDate")
                    if not date_tag: continue
                    date_str = await date_tag.inner_text()
                    
                    try:
                        dt = datetime.strptime(date_str.strip(), '%m/%d/%Y')
                    except:
                        continue
                        
                    if dt < cutoff_date: continue
                    
                    title_link = await item.query_selector(".itemTitle a")
                    if not title_link: continue
                    
                    href = await title_link.get_attribute("href")
                    title = await title_link.inner_text()
                    
                    speaker_tag = await item.query_selector(".news__speaker")
                    speaker = await speaker_tag.inner_text() if speaker_tag else 'N/A'
                    
                    speech_links.append({
                        'url': base_url + href if href.startswith('/') else href,
                        'date': dt.strftime('%Y-%m-%d'),
                        'title': title.strip(),
                        'speaker': speaker.strip()
                    })
                except Exception as e:
                    logger.error(f"Error parsing Fed list item: {e}")
            
            await browser.close()

        for link_info in speech_links:
            content = self._scrape_fed_content(link_info['url'])
            if content:
                link_info.update(content)
                link_info.update({
                    'source': 'Fed',
                    'country': 'USA'
                })
                speeches.append(link_info)
            time.sleep(1)
            
        return self.save_speeches(speeches)

    # --- Mann Ki Baat Logic ---

    def _ordinal(self, n):
        """Return ordinal string for n, e.g. 1 -> '1st'"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    def scrape_mann_ki_baat(self, transcripts_dir='./mann_ki_baat_transcripts'):
        """
        Load Mann Ki Baat transcripts from already-scraped local .txt files.
        Falls back to web scraping for any missing episodes.
        File format: first line = "Episode N (DD Mon, YYYY)", rest = transcript.
        """
        logger.info(f"Loading Mann Ki Baat transcripts from: {transcripts_dir}")
        speeches = []

        if not os.path.isdir(transcripts_dir):
            logger.warning(f"Transcript directory not found: {transcripts_dir}")
            return 0

        import glob
        txt_files = sorted(glob.glob(os.path.join(transcripts_dir, 'mann_ki_baat_*.txt')))
        logger.info(f"Found {len(txt_files)} local MKB transcript files.")

        for fpath in txt_files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                header = lines[0].strip()   # e.g. "Episode 80 (29 Aug, 2021)"
                full_text = '\n'.join(lines[2:]).strip()  # skip blank line after header

                # Parse episode number
                ep_match = re.match(r'Episode\s+(\d+)', header)
                ep_num = int(ep_match.group(1)) if ep_match else None

                # Parse date from header like "(29 Aug, 2021)" or "(29 Aug 2021)"
                parsed_date = None
                date_match = re.search(r'\(([^)]+)\)', header)
                if date_match:
                    date_str = date_match.group(1).strip()
                    for fmt in ['%d %b, %Y', '%d %B, %Y', '%d %b %Y', '%d %B %Y',
                                '%B %d, %Y', '%b %d, %Y']:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue

                speeches.append({
                    'date': parsed_date,
                    'source': 'Mann Ki Baat',
                    'country': 'India',
                    'speaker': 'PM Modi',
                    'title': f"Mann Ki Baat - Episode {ep_num}" if ep_num else header,
                    'full_text': full_text,
                    'url': f"https://www.pmindia.gov.in/en/news_updates/pms-address-in-the-{self._ordinal(ep_num)}-episode-of-mann-ki-baat/" if ep_num else None
                })
                logger.info(f"Loaded MKB Episode {ep_num} ({parsed_date})")

            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")

        logger.info(f"Mann Ki Baat: loaded {len(speeches)} episodes from local files.")
        return self.save_speeches(speeches)

    async def scrape_all(self, days_back=365):
        """Scrape all sources: ECB, Fed and Mann Ki Baat"""
        self.scrape_mann_ki_baat()  # Removed max_episodes to match signature
        await self.scrape_ecb(days_back=days_back)
        await self.scrape_fed(days_back=days_back)

    def _scrape_fed_content(self, url):
        """Scrapes individual Fed speech content"""
        try:
            response = requests.get(url, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content_div = soup.find('div', class_='col-xs-12 col-sm-8') or soup.find('div', id='content')
            if not content_div: return None
            
            # Remove unwanted
            for unwanted in content_div.find_all(['script', 'style', 'nav']):
                unwanted.decompose()
                
            paragraphs = content_div.find_all('p')
            speech_text = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50 and not re.search(r'last update:|return to text|share', text, re.I):
                    speech_text.append(text)
            
            return {'full_text': '\n\n'.join(speech_text)}
        except Exception as e:
            logger.error(f"Error scraping Fed content {url}: {e}")
            return None

if __name__ == "__main__":
    scraper = CentralizedSpeechScraper()
    # To run: python centralized_scraper.py
    asyncio.run(scraper.scrape_ecb(days_back=30))
    scraper.scrape_fed(days_back=30)
