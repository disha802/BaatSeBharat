import asyncio
from playwright.async_api import async_playwright
from datetime import datetime, timedelta

async def debug_fed():
    url = "https://www.federalreserve.gov/newsevents/speeches.htm"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"Loading {url}...")
        try:
            await page.goto(url, wait_until="load", timeout=30000)
            await asyncio.sleep(5) # Wait for JS to render
            
            # Wait for any .row or .itemTitle
            try:
                await page.wait_for_selector(".itemTitle", timeout=10000)
            except:
                print("Warning: Timeout waiting for .itemTitle")
            
            # Extract links
            links = await page.evaluate("Array.from(document.querySelectorAll('a')).map(a => a.href)")
            print(f"Found {len(links)} total links.")
            
            relevant_links = [l for l in links if l and '/newsevents/speech/' in l]
            print(f"Found {len(relevant_links)} links with '/newsevents/speech/'.")
            for l in relevant_links[:20]:
                print(f" - {l}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_fed())
