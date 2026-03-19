import requests
from bs4 import BeautifulSoup
import os

# Function to get ordinal suffix
def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:  # 11th, 12th, 13th...
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# Folder to store transcripts
os.makedirs("mann_ki_baat_transcripts", exist_ok=True)

# Base URL pattern
base_url = "https://www.pmindia.gov.in/en/news_updates/pms-address-in-the-{}-episode-of-mann-ki-baat/"

# Loop through episode numbers
for i in range(1, 126):  # 1 to 125
    try:
        episode_str = ordinal(i)
        url = base_url.format(episode_str)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Skipping Episode {i} ({url}) - not found")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract date
        date_tag = soup.find("span", class_="date")
        date_text = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

        # Extract transcript paragraphs
        news_bg = soup.find("div", class_="news-bg")
        if not news_bg:
            print(f"⚠️ No transcript found for Episode {i}")
            continue

        paragraphs = [p.get_text(strip=True) for p in news_bg.find_all("p")]
        transcript_text = "\n".join(paragraphs)

        # Save to file
        file_path = os.path.join("mann_ki_baat_transcripts", f"mann_ki_baat_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Episode {i} ({date_text})\n\n")
            f.write(transcript_text)

        print(f"✅ Saved Episode {i}")

    except Exception as e:
        print(f"⚠️ Error on Episode {i}: {e}")