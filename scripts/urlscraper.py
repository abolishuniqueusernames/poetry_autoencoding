
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Optional

def extract_hyperlinks(url: str, name: str, timeout: int = 10) -> List[str]:
    """
    Extract all hyperlinks from a webpage.
    
    Args:
        url: The URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        List of absolute URLs found on the page
    """
    try:
        # Add headers to appear more like a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # Specify encoding if it's not detected correctly
        if response.encoding is None:
            response.encoding = 'utf-8'
            
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []
    
    links = []
    for tag in soup.find_all('a', href=True):
        href = tag.get('href', '').strip()
        if href:  # Skip empty hrefs
            full_url = urljoin(url, href)
            links.append(full_url)

    print(f"Found {len(links)} links. Writing to file")
    with open(f"hyperlinks_{name}_readable.txt", 'w', encoding='utf-8') as f:
        for link in links:
            f.write(f"\'{link}\',\n")
    
    return links
# Actually capture and display the results
if __name__ == "__main__":
    links = extract_hyperlinks('https://www.dreamboybook.club/archive','dreamboyarchive')
    links = extract_hyperlinks('https://www.dreamboybook.club/collected-works','dreamboyarchive2')
   
