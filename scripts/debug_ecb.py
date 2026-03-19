import asyncio
from playwright.async_api import async_playwright
import re
from datetime import datetime, timedelta

async def debug_ecb():
    url = "https://www.ecb.europa.eu/press/key/html/index.en.html"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"Loading {url}...")
        try:
            await page.goto(url, wait_until="load", timeout=30000)
            await asyncio.sleep(5) # Wait for JS to render
            
            # Scroll
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
                
            links = await page.evaluate("Array.from(document.querySelectorAll('a')).map(a => a.href)")
            print(f"Found {len(links)} total links.")
            
            relevant_links = [l for l in links if l and '/press/key/date/' in l]
            print(f"Found {len(relevant_links)} links with '/press/key/date/'.")
            for l in relevant_links[:20]:
                print(f" - {l}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_ecb())
