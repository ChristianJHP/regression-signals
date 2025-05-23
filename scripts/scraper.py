

import requests
import time
import datetime
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

def get_cmc_trending():
    url = "https://coinmarketcap.com/trending-cryptocurrencies/"
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("table tbody tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) > 2:
            name = cols[2].text.strip()
            symbol = cols[2].find("p").text.strip()
            data.append({
                "source": "CMC_Trending",
                "name": name,
                "symbol": symbol,
                "timestamp": datetime.datetime.utcnow()
            })
    return data

def get_dextools_gainers():
    # Placeholder: Real DEXTools data requires API or scraping JavaScript-rendered data.
    return []

def scrape_all():
    cmc_data = get_cmc_trending()
    dextools_data = get_dextools_gainers()
    combined = cmc_data + dextools_data
    return pd.DataFrame(combined)

if __name__ == "__main__":
    df = scrape_all()
    if not df.empty:
        filename = f"data/pump_candidates_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} pump candidates to {filename}")
    else:
        print("No pump candidates found.")