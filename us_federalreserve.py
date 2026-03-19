from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import requests
from bs4 import BeautifulSoup  # This imports from beautifulsoup4
import os
import sys

class FederalReserveFullContentScraper:
    def __init__(self):
        """Initialize the scraper with configuration"""
        # Chrome options for headless browsing
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')  # Run in background
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.driver = None
        self.base_url = "https://www.federalreserve.gov"
        self.speeches = []
        
        # Calculate date range: current date to one year ago
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365)
        
        print("="*60)
        print("FEDERAL RESERVE FULL CONTENT SCRAPER")
        print("="*60)
        print(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print("="*60)
    
    def setup_driver(self):
        """Initialize the Chrome driver"""
        try:
            self.driver = webdriver.Chrome(options=self.chrome_options)
            return True
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("\nPlease ensure ChromeDriver is installed:")
            print("1. Download from: https://chromedriver.chromium.org/")
            print("2. Add it to your PATH or place in current directory")
            return False
    
    def extract_speech_links_from_container(self, container):
        """Extract individual speech links from a container"""
        links = []
        
        # Find all links in the container
        all_links = container.find_elements(By.TAG_NAME, "a")
        
        for link in all_links:
            href = link.get_attribute('href')
            link_text = link.text.strip()
            
            # Filter for actual speech links (not navigation)
            if href and ('/newsevents/speech/' in href or '/newsevents/testimony/' in href):
                links.append({
                    'url': href,
                    'text': link_text
                })
        
        return links
    
    def scrape_full_speech_content(self, url):
        """
        Extract the full text content from an individual speech page
        This is the key function that gets the actual speech text
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"  Failed to fetch {url}: Status {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # METHOD 1: Try to find the main content container
            content_containers = [
                ('div', {'class': 'col-xs-12 col-sm-8'}),
                ('div', {'class': 'col-xs-12 col-sm-9'}),
                ('div', {'id': 'content'}),
                ('article', {}),
                ('div', {'class': 'panel-body'}),
                ('div', {'class': 'row'}),
            ]
            
            content = None
            for tag, attrs in content_containers:
                if attrs:
                    content = soup.find(tag, attrs=attrs)
                else:
                    content = soup.find(tag)
                if content:
                    print(f"  Found content container: {tag} with {attrs}")
                    break
            
            # METHOD 2: If no container found, get all paragraphs
            if not content:
                print("  No specific container found, extracting all paragraphs...")
                paragraphs = soup.find_all('p')
                speech_paragraphs = []
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    # Include longer paragraphs that look like speech content
                    if (len(text) > 100 and 
                        not re.search(r'last update|return to text|share|media advisory', text, re.I)):
                        speech_paragraphs.append(text)
                
                if speech_paragraphs:
                    return '\n\n'.join(speech_paragraphs)
                return None
            
            # METHOD 3: Clean and extract from found container
            # Remove unwanted elements
            for unwanted in content.find_all(['script', 'style', 'button', 'footer', 'nav']):
                unwanted.decompose()
            
            # Get all paragraphs within the content
            paragraphs = content.find_all('p')
            speech_text = []
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                # Filter out metadata and short lines
                if (len(text) > 50 and 
                    not re.search(r'last update:|return to text|share|media advisory|^share$|^watch live$', text, re.I) and
                    not text.startswith('Last Update') and
                    not text.startswith('Return to text')):
                    speech_text.append(text)
            
            if speech_text:
                return '\n\n'.join(speech_text)
            
            # METHOD 4: If no paragraphs, get all text
            all_text = content.get_text(separator='\n', strip=True)
            # Clean up excessive newlines
            all_text = re.sub(r'\n\s*\n', '\n\n', all_text)
            
            # Remove metadata lines
            lines = all_text.split('\n')
            filtered_lines = []
            for line in lines:
                if not re.search(r'last update|return to text|share|media advisory', line, re.I):
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines) if filtered_lines else None
            
        except Exception as e:
            print(f"  Error scraping {url}: {e}")
            return None
    
    def parse_speeches_from_page(self):
        """Parse the main speeches page and extract all speeches with full content"""
        print("\nAnalyzing page for speech content...")
        
        # Wait for dynamic content to load
        time.sleep(5)
        
        # Try to find the main content area that contains speeches
        possible_containers = []
        
        # Look for divs that might contain the speech listing
        selectors = [
            "div.row",
            "div.col-xs-12",
            "div.panel",
            "div.content",
            "main",
            "article",
            "div#content"
        ]
        
        for selector in selectors:
            elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
            for elem in elements:
                text = elem.text
                if text and any(name in text for name in ['Governor', 'Vice Chair', 'Chair']):
                    possible_containers.append(elem)
                    print(f"  Found potential container with selector: {selector}")
                    break
        
        if not possible_containers:
            print("  No speech containers found")
            return False
        
        print(f"\nFound {len(possible_containers)} potential speech containers")
        
        # Process each container
        all_speeches_data = []
        
        for container in possible_containers:
            # Get all text lines
            text = container.text
            lines = text.split('\n')
            
            # Get all speech links in this container
            speech_links = self.extract_speech_links_from_container(container)
            link_index = 0
            
            i = 0
            while i < len(lines) - 1:
                current_line = lines[i].strip()
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                
                # Check if this line contains a speaker
                if any(title in current_line for title in ['Governor', 'Vice Chair', 'Chair']):
                    speaker = current_line
                    description = next_line
                    
                    # Extract year from description
                    year_match = re.search(r'\b(20\d{2})\b', description)
                    year = int(year_match.group(1)) if year_match else None
                    
                    # Get the corresponding link
                    link_url = None
                    if link_index < len(speech_links):
                        link_url = speech_links[link_index]['url']
                        link_index += 1
                    
                    # Only include speeches from target years
                    if year and year >= 2025:
                        print(f"\n  Processing: {speaker[:50]}...")
                        print(f"  Year: {year}")
                        print(f"  Link: {link_url}")
                        
                        # Scrape the full content
                        content = None
                        if link_url and ('/newsevents/speech/' in link_url or '/newsevents/testimony/' in link_url):
                            print(f"  Fetching full speech content...")
                            content = self.scrape_full_speech_content(link_url)
                            
                            # Show content preview
                            if content:
                                preview = content[:200].replace('\n', ' ')
                                print(f"  ✓ Content obtained ({len(content)} chars)")
                                print(f"  Preview: {preview}...")
                            else:
                                print(f"  ✗ No content extracted")
                            
                            time.sleep(1)  # Be respectful to the server
                        
                        # Create speech entry
                        speech_entry = {
                            'year': year,
                            'speaker': speaker,
                            'description': description,
                            'link': link_url,
                            'content': content,
                            'content_length': len(content) if content else 0,
                            'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        all_speeches_data.append(speech_entry)
                    
                    i += 2  # Skip the description line
                else:
                    i += 1
        
        # Remove duplicates based on link (some speeches might appear twice)
        seen_links = set()
        for speech in all_speeches_data:
            if speech['link'] not in seen_links:
                seen_links.add(speech['link'])
                self.speeches.append(speech)
        
        return len(self.speeches) > 0
    
    def save_results(self):
        """Save the scraped data to CSV and JSON files"""
        if not self.speeches:
            print("\nNo speeches to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.speeches)
        
        # Remove the temporary content_length column if it exists
        if 'content_length' in df.columns:
            df = df.drop('content_length', axis=1)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'fed_speeches_full_{timestamp}.csv'
        json_filename = f'fed_speeches_full_{timestamp}.json'
        
        # Save to CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\n✓ Saved {len(self.speeches)} speeches to {csv_filename}")
        
        # Save to JSON for better content preservation
        df.to_json(json_filename, orient='records', indent=2)
        print(f"✓ Saved to {json_filename}")
        
        # Print detailed summary
        print("\n" + "="*60)
        print("SCRAPING SUMMARY")
        print("="*60)
        print(f"Total unique speeches: {len(self.speeches)}")
        
        speeches_with_content = sum(1 for s in self.speeches if s['content'])
        print(f"Speeches with full content: {speeches_with_content}")
        
        if speeches_with_content > 0:
            avg_length = sum(len(s['content']) for s in self.speeches if s['content']) // speeches_with_content
            print(f"Average content length: {avg_length:,} characters")
        
        # Yearly breakdown
        print("\nYearly breakdown:")
        years = {}
        for speech in self.speeches:
            year = speech['year']
            years[year] = years.get(year, 0) + 1
        
        for year in sorted(years.keys()):
            with_content = sum(1 for s in self.speeches if s['year'] == year and s['content'])
            print(f"  {year}: {years[year]} speeches ({with_content} with content)")
        
        # Show sample entries
        print("\n" + "="*60)
        print("SAMPLE ENTRIES (with content preview)")
        print("="*60)
        
        for i, speech in enumerate(self.speeches[:3]):
            print(f"\n{i+1}. {speech['year']} - {speech['speaker']}")
            print(f"   Description: {speech['description'][:100]}...")
            if speech['content']:
                preview = speech['content'][:200].replace('\n', ' ')
                print(f"   Content preview: {preview}...")
            else:
                print(f"   Content: Not available")
            print(f"   Link: {speech['link']}")
    
    def run(self):
        """Main execution method"""
        # Setup Chrome driver
        if not self.setup_driver():
            print("\nFailed to setup Chrome driver. Exiting.")
            return
        
        try:
            # Navigate to the speeches page
            url = "https://www.federalreserve.gov/newsevents/speeches-testimony.htm"
            print(f"\nNavigating to: {url}")
            self.driver.get(url)
            
            # Wait for page to load
            print("Waiting for dynamic content to load...")
            time.sleep(5)
            
            # Parse speeches and get full content
            if self.parse_speeches_from_page():
                self.save_results()
            else:
                print("\nNo speeches found. Saving debug information...")
                
                # Save screenshot and HTML for debugging
                self.driver.save_screenshot('debug_screenshot.png')
                with open('debug_page.html', 'w', encoding='utf-8') as f:
                    f.write(self.driver.page_source)
                print("Saved debug files: debug_screenshot.png and debug_page.html")
        
        finally:
            # Clean up
            if self.driver:
                self.driver.quit()
        
        print("\n" + "="*60)
        print("SCRAPING COMPLETE")
        print("="*60)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'selenium': 'selenium',
        'pandas': 'pandas', 
        'requests': 'requests',
        'bs4': 'beautifulsoup4'  # Note: bs4 is the import name, beautifulsoup4 is the package name
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\nMissing required packages. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Checking dependencies...")
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing packages and try again.")
        exit(1)
    
    print("All dependencies satisfied!")
    
    # Run the scraper
    scraper = FederalReserveFullContentScraper()
    scraper.run()