import asyncio
from playwright.async_api import async_playwright
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re

async def get_all_speeches_playwright(days_back=365):
    """
    Uses Playwright to handle dynamically loaded content and expand all sections
    """
    cutoff_date = datetime.now() - timedelta(days=days_back)
    speech_urls = set()
    
    url = "https://www.ecb.europa.eu/press/key/html/index.en.html"
    
    async with async_playwright() as p:
        print("🚀 Launching browser...")
        browser = await p.chromium.launch(headless=False)  # Set to False to see what's happening
        page = await browser.new_page()
        
        print(f"🔍 Loading speeches page: {url}")
        await page.goto(url, wait_until="networkidle")
        
        # Handle cookie consent
        try:
            accept_button = await page.query_selector("button:has-text('Accept')")
            if accept_button:
                await accept_button.click()
                print("🍪 Accepted cookies")
                await page.wait_for_timeout(2000)
        except:
            print("🍪 No cookie consent needed")
        
        # Look for and clear any filters
        print("🔍 Looking for filters to clear...")
        
        # Try to find and click "All" or "Clear filters" buttons
        try:
            # Look for filter dropdowns or "All" options
            all_buttons = await page.query_selector_all("button:has-text('All'), a:has-text('All')")
            for btn in all_buttons:
                await btn.click()
                print("   Clicked 'All' button")
                await page.wait_for_timeout(1000)
        except:
            print("   No 'All' buttons found")
        
        # Try to expand all year sections
        print("🔍 Expanding all year sections...")
        
        # Look for collapsible sections (common pattern: year headings)
        year_sections = await page.query_selector_all("div.year, section.year, details, summary")
        
        if year_sections:
            print(f"   Found {len(year_sections)} potential sections")
            for section in year_sections:
                try:
                    await section.click()
                    print("   Expanded a section")
                    await page.wait_for_timeout(500)
                except:
                    pass
        else:
            print("   No obvious sections found, trying alternative method...")
            # Try to find and click all expandable elements
            expand_buttons = await page.query_selector_all("button[aria-expanded], .expand, .toggle")
            for btn in expand_buttons:
                try:
                    await btn.click()
                    print("   Clicked expand button")
                    await page.wait_for_timeout(500)
                except:
                    pass
        
        # Now scroll to load everything
        print("🔄 Scrolling to load all content...")
        
        last_height = await page.evaluate("document.body.scrollHeight")
        scroll_count = 0
        no_change_count = 0
        
        while no_change_count < 3 and scroll_count < 10:  # Cap at 10 scrolls for safety
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            scroll_count += 1
            await page.wait_for_timeout(3000)  # Longer wait for content to load
            
            new_height = await page.evaluate("document.body.scrollHeight")
            print(f"   Scroll {scroll_count}: height {new_height}")
            
            if new_height == last_height:
                no_change_count += 1
                print(f"      No change ({no_change_count}/3)")
            else:
                no_change_count = 0
                
            last_height = new_height
        
        print(f"📜 Reached end after {scroll_count} scrolls")
        
        # Take a screenshot to debug
        await page.screenshot(path="ecb_page_debug.png")
        print("📸 Saved debug screenshot as 'ecb_page_debug.png'")
        
        # Get all links
        print("🔗 Extracting all links...")
        links = await page.evaluate("""
            Array.from(document.querySelectorAll('a')).map(a => a.href)
        """)
        
        # Also try to get text to see what years are present
        page_text = await page.evaluate("document.body.innerText")
        print(f"📝 Page contains years: {re.findall(r'20\d{2}', page_text)[:10]}")
        
        await browser.close()
        
        # Filter for speeches
        print(f"📊 Found {len(links)} total links, filtering for speeches...")
        
        speech_count = 0
        for href in links:
            if not href or not isinstance(href, str):
                continue
            
            if '/press/key/date/' in href and 'sp' in href and href.endswith('.en.html'):
                # New pattern: ecb.sp260309~6cfdbd02b7.en.html or old pattern: sp260309.en.html
                date_match = re.search(r'sp(\d{2})(\d{2})(\d{2})', href)
                if date_match:
                    year, month, day = date_match.groups()
                    pub_year = 2000 + int(year)
                    pub_date = datetime(pub_year, int(month), int(day))
                    
                    if pub_date >= cutoff_date:
                        speech_urls.add(href)
                        speech_count += 1
                        print(f"   ✓ {speech_count}: {pub_date.strftime('%Y-%m-%d')} - {href.split('/')[-1]}")
        
        print(f"\n✅ Found {len(speech_urls)} speeches from last {days_back} days")
        return list(speech_urls)

def scrape_speech_content(url):
    """Scrapes individual speech content"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Title
        title_tag = soup.find('h1', class_='title') or soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else 'N/A'

        # Date
        date_tag = soup.find('p', class_='date')
        date_str = date_tag.get_text(strip=True) if date_tag else 'N/A'

        # Speaker
        speaker = 'N/A'
        subtitle_tag = soup.find('p', class_='subtitle')
        if subtitle_tag:
            speaker_text = subtitle_tag.get_text(strip=True)
            if 'by' in speaker_text:
                speaker = speaker_text.split('by')[-1].split(',')[0].strip()

        # Full text
        content_div = soup.find('div', class_='ecb-pressContent') or soup.find('main')
        full_text = ""
        if content_div:
            paragraphs = content_div.find_all('p')
            for p in paragraphs:
                if not p.find_parent('div', class_='contact'):
                    full_text += p.get_text(strip=True) + "\n\n"

        return {
            'url': url,
            'title': title,
            'date': date_str,
            'speaker': speaker,
            'full_text': full_text.strip()
        }

    except Exception as e:
        print(f"    ❌ Error scraping {url.split('/')[-1]}: {e}")
        return None

async def main():
    print("=" * 70)
    print("ECB Speech Scraper - Last 365 Days")
    print("=" * 70)
    
    # Get all URLs
    print("\n🌐 Step 1: Discovering speeches...")
    speech_urls = await get_all_speeches_playwright(days_back=365)
    
    if not speech_urls:
        print("\n❌ No speeches found.")
        return
    
    print(f"\n📊 Step 2: Scraping {len(speech_urls)} speeches...")
    print(f"⏱️  This will take about {len(speech_urls)} seconds\n")
    
    # Scrape each speech
    all_speeches = []
    for i, url in enumerate(speech_urls):
        print(f"  {i+1:3d}/{len(speech_urls)}: {url.split('/')[-1]}")
        speech_data = scrape_speech_content(url)
        if speech_data:
            all_speeches.append(speech_data)
        time.sleep(1)
    
    # Save results
    if all_speeches:
        df = pd.DataFrame(all_speeches)
        
        # Sort by date
        try:
            df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('parsed_date', ascending=False)
        except:
            pass
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"ecb_speeches_{timestamp}.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 70)
        print(f"✅ SUCCESS! Scraped {len(all_speeches)} speeches")
        print(f"📁 Saved to: {output_filename}")
        print("=" * 70)
        
        # Show sample
        print("\n📋 First 10 speeches:")
        for _, row in df.head(10).iterrows():
            date = str(row.get('date', 'N/A'))[:10]
            print(f"  • {date} - {row.get('speaker', 'N/A')}: {row.get('title', 'N/A')[:60]}...")
    else:
        print("\n❌ No speeches were successfully scraped")

if __name__ == "__main__":
    # Install playwright if needed
    import subprocess
    import sys
    
    try:
        import playwright
        print("✅ Playwright already installed")
    except:
        print("📦 Installing Playwright...")
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
        print("📦 Installing Playwright browsers...")
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        print("✅ Playwright installed")
    
    # Run the scraper
    asyncio.run(main())