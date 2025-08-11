import time
import json
import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class AltLitPoetryScraper:
    def __init__(self, headless=True):
        self.poems_found = []
        self.setup_firefox_driver(headless)
        
    def setup_firefox_driver(self, headless=True):
        """Setup Firefox with enhanced configuration for alt-lit sites"""
        try:
            firefox_options = FirefoxOptions()
            if headless:
                firefox_options.add_argument("--headless")
            
            # Enhanced user agent and preferences for modern sites
            firefox_options.set_preference("general.useragent.override", 
                "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0")
            
            # Better JavaScript support
            firefox_options.set_preference("javascript.enabled", True)
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            
            # Handle modern web features
            firefox_options.set_preference("network.http.use-cache", False)
            firefox_options.set_preference("browser.cache.disk.enable", False)
            firefox_options.set_preference("browser.cache.memory.enable", False)
            
            self.driver = webdriver.Firefox(options=firefox_options)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, 20)
            
            print("✓ Firefox WebDriver setup successful")
            
        except Exception as e:
            print(f"✗ Firefox setup failed: {e}")
            print("Make sure geckodriver is installed: https://github.com/mozilla/geckodriver/releases")
            raise
    
    def test_firefox_functionality(self):
        """Test basic Firefox functionality"""
        try:
            print("Testing Firefox functionality...")
            self.driver.get("https://httpbin.org/user-agent")
            time.sleep(2)
            
            # Check if we can access the page
            body = self.driver.find_element(By.TAG_NAME, "body")
            if "Firefox" in body.text:
                print("✓ Firefox is working correctly")
                return True
            else:
                print("✗ Firefox user agent issue")
                return False
                
        except Exception as e:
            print(f"✗ Firefox test failed: {e}")
            return False
    
    def scrape_university_mfa_repositories(self):
        """Target university repositories with contemporary MFA work"""
        
        # Universities known for strong creative writing programs
        mfa_repositories = [
            {
                'name': 'Iowa Writers Workshop Repository',
                'search_url': 'https://ir.uiowa.edu/cgi/search.cgi?q=poetry&context=all&type=&sort=date',
                'base_url': 'https://ir.uiowa.edu',
                'link_pattern': r'/etd/\d+'
            },
            {
                'name': 'Syracuse Creative Writing',
                'search_url': 'https://surface.syr.edu/do/search/?q=poetry&start=0&context=509135',
                'base_url': 'https://surface.syr.edu',
                'link_pattern': r'/etd/\d+'
            },
            {
                'name': 'Columbia University Academic Commons',
                'search_url': 'https://academiccommons.columbia.edu/search?f%5Bgenre_facet%5D%5B%5D=Creative+writing&q=poetry',
                'base_url': 'https://academiccommons.columbia.edu',
                'link_pattern': r'/doi/\d+'
            },
            {
                'name': 'NYU Faculty Digital Archive',
                'search_url': 'https://archive.nyu.edu/search?f%5Bresource_type%5D%5B%5D=Creative+Work&q=poetry',
                'base_url': 'https://archive.nyu.edu',
                'link_pattern': r'/handle/\d+'
            }
        ]
        
        all_poems = []
        
        for repo in mfa_repositories:
            try:
                print(f"\n=== SCRAPING {repo['name']} ===")
                poems = self.scrape_repository_search(repo)
                all_poems.extend(poems)
                print(f"Found {len(poems)} poems from {repo['name']}")
                
            except Exception as e:
                print(f"Error with {repo['name']}: {e}")
        
        return all_poems
    
    def scrape_repository_search(self, repo_config):
        """Generic repository scraper with enhanced debugging"""
        poems = []
        
        try:
            print(f"Loading search page: {repo_config['search_url']}")
            self.driver.get(repo_config['search_url'])
            
            # Wait for page to load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(3)
            
            # Debug: Log page title and basic info
            title = self.driver.title
            print(f"Page title: {title}")
            
            # Find all links that match the pattern
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            matching_links = []
            
            for link in all_links:
                href = link.get_attribute('href')
                if href and re.search(repo_config['link_pattern'], href):
                    text = link.text.strip()
                    if len(text) > 5:  # Has meaningful text
                        matching_links.append({
                            'url': href,
                            'title': text[:100],  # Truncate long titles
                            'full_text': text
                        })
            
            print(f"Found {len(matching_links)} potential poetry links")
            
            # Extract from first few links (be respectful)
            for link_info in matching_links[:10]:  
                try:
                    poem_data = self.extract_contemporary_poem(
                        link_info['url'], 
                        link_info['title'],
                        repo_config['name']
                    )
                    
                    if poem_data:
                        poems.append(poem_data)
                        print(f"✓ Extracted: {poem_data['title'][:50]}...")
                    
                    time.sleep(2)  # Be respectful
                    
                except Exception as e:
                    print(f"✗ Failed to extract from {link_info['url']}: {e}")
                    continue
            
        except Exception as e:
            print(f"Repository scraping failed: {e}")
        
        return poems
    
    def extract_contemporary_poem(self, url, title, source):
        """Extract poem with focus on contemporary/alt-lit style detection"""
        try:
            print(f"Extracting from: {url}")
            self.driver.get(url)
            time.sleep(3)
            
            # Look for content in order of likelihood
            content_selectors = [
                # Academic repository patterns
                '.simple-item-view-description',
                '.ds-metadata-field-value',
                '.artifact-description',
                '.item-summary',
                
                # Generic content patterns
                'div[class*="content"]',
                'div[class*="abstract"]',
                'div[class*="description"]',
                'div[class*="text"]',
                'section[class*="content"]',
                
                # Text containers
                'pre', 'blockquote', 'p',
                
                # Academic specific
                '.mods-abstract',
                '.dc-description',
                '.metadata-field'
            ]
            
            for selector in content_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        text = element.text.strip()
                        
                        if self.looks_like_contemporary_poetry(text):
                            return {
                                'title': title,
                                'text': text,
                                'url': url,
                                'source': source,
                                'selector_used': selector,
                                'length': len(text),
                                'line_count': len(text.split('\n')),
                                'style_indicators': self.analyze_alt_lit_style(text)
                            }
                
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            print(f"Extraction error: {e}")
            return None
    
    def looks_like_contemporary_poetry(self, text):
        """Enhanced detection for contemporary/alt-lit poetry"""
        if len(text) < 30 or len(text) > 8000:
            return False
        
        # Skip obvious academic metadata
        academic_keywords = [
            'Abstract:', 'Keywords:', 'Bibliography', 'References',
            'Advisor:', 'Committee:', 'University', 'Thesis',
            'Copyright', 'All rights reserved', 'Published by',
            'DOI:', 'ISSN:', 'Volume', 'Issue'
        ]
        
        if any(keyword in text for keyword in academic_keywords):
            return False
        
        # Look for poetry characteristics
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
        
        # Contemporary poetry indicators
        contemporary_indicators = [
            # Alt-lit style markers
            len([line for line in lines if line.islower()]) > len(lines) * 0.3,  # Lots of lowercase
            'i ' in text.lower(),  # First person lowercase
            text.count('\n\n') > 0,  # Stanza breaks
            any(char in text for char in ['@', '#', 'http']),  # Digital elements
            
            # Modern themes/language
            any(word in text.lower() for word in [
                'internet', 'phone', 'text', 'instagram', 'twitter', 'app',
                'anxiety', 'depression', 'uber', 'netflix', 'wifi', 'laptop',
                'millennial', 'gen z', 'dating app', 'social media'
            ]),
            
            # Poetic structure
            10 < sum(len(line) for line in lines) / len(lines) < 80,  # Line length
            len(lines) > 4,  # Multiple lines
        ]
        
        # Must have at least 2 contemporary indicators
        if sum(contemporary_indicators) >= 2:
            return True
        
        # Alternative: traditional poetry structure but contemporary language
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if 15 < avg_line_length < 60 and len(lines) > 6:
            modern_words = ['like', 'just', 'really', 'maybe', 'kinda', 'sorta']
            if any(word in text.lower() for word in modern_words):
                return True
        
        return False
    
    def analyze_alt_lit_style(self, text):
        """Analyze how much text matches alt-lit aesthetic"""
        indicators = {
            'lowercase_lines': len([line for line in text.split('\n') if line and line.islower()]),
            'first_person': text.lower().count('i '),
            'modern_references': sum(1 for word in ['internet', 'phone', 'app', 'wifi', 'laptop', 'uber'] if word in text.lower()),
            'emotional_words': sum(1 for word in ['anxiety', 'depression', 'lonely', 'sad', 'empty'] if word in text.lower()),
            'casual_language': sum(1 for word in ['like', 'just', 'really', 'kinda', 'sorta', 'whatever'] if word in text.lower()),
            'stanza_breaks': text.count('\n\n'),
            'total_lines': len([line for line in text.split('\n') if line.strip()])
        }
        
        return indicators
    
    def scrape_contemporary_lit_magazines(self):
        """Scrape digital-first literary magazines"""
        
        # Digital literary magazines with alt-lit content
        digital_magazines = [
            {
                'name': 'Hobart Pulp',
                'url': 'https://www.hobartpulp.com/poetry',
                'content_selector': '.entry-content'
            },
            {
                'name': 'Shabby Doll House',
                'url': 'https://shabbydollhouse.com/poetry',
                'content_selector': '.post-content'
            },
            {
                'name': 'Electric Literature',
                'url': 'https://electricliterature.com/tag/poetry/',
                'content_selector': '.article-content'
            }
        ]
        
        poems = []
        
        for magazine in digital_magazines:
            try:
                print(f"\n=== SCRAPING {magazine['name']} ===")
                mag_poems = self.scrape_magazine_poetry(magazine)
                poems.extend(mag_poems)
                
            except Exception as e:
                print(f"Error scraping {magazine['name']}: {e}")
        
        return poems
    
    def scrape_magazine_poetry(self, magazine_config):
        """Scrape poetry from digital magazines"""
        poems = []
        
        try:
            self.driver.get(magazine_config['url'])
            time.sleep(3)
            
            # Find article/post links
            article_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="poem"], a[href*="poetry"]')
            
            for link in article_links[:5]:  # Limit scraping
                try:
                    article_url = link.get_attribute('href')
                    article_title = link.text.strip()
                    
                    if len(article_title) > 5:
                        poem_data = self.extract_magazine_poem(
                            article_url, 
                            article_title, 
                            magazine_config
                        )
                        
                        if poem_data:
                            poems.append(poem_data)
                            print(f"✓ Found poem: {poem_data['title'][:30]}...")
                    
                    time.sleep(2)
                    
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"Magazine scraping error: {e}")
        
        return poems
    
    def extract_magazine_poem(self, url, title, magazine_config):
        """Extract poem from magazine article"""
        try:
            self.driver.get(url)
            time.sleep(2)
            
            content_element = self.driver.find_element(By.CSS_SELECTOR, magazine_config['content_selector'])
            text = content_element.text.strip()
            
            if self.looks_like_contemporary_poetry(text):
                return {
                    'title': title,
                    'text': text,
                    'url': url,
                    'source': magazine_config['name'],
                    'length': len(text),
                    'line_count': len(text.split('\n')),
                    'style_indicators': self.analyze_alt_lit_style(text)
                }
            
            return None
            
        except Exception as e:
            print(f"Magazine extraction error: {e}")
            return None
    
    def run_alt_lit_scraping_session(self):
        """Run comprehensive alt-lit poetry scraping"""
        print("=== ALT-LIT POETRY SCRAPING SESSION ===\n")
        
        # Test Firefox first
        if not self.test_firefox_functionality():
            print("Firefox test failed. Please check your setup.")
            return []
        
        all_poems = []
        
        # Scrape university repositories (contemporary MFA work)
        print("\n=== PHASE 1: UNIVERSITY MFA REPOSITORIES ===")
        university_poems = self.scrape_university_mfa_repositories()
        all_poems.extend(university_poems)
        
        # Scrape digital literary magazines
        print("\n=== PHASE 2: DIGITAL LITERARY MAGAZINES ===")
        magazine_poems = self.scrape_contemporary_lit_magazines()
        all_poems.extend(magazine_poems)
        
        # Results
        print(f"\n=== SCRAPING COMPLETE ===")
        print(f"Total poems found: {len(all_poems)}")
        
        if all_poems:
            # Analyze the collection
            self.analyze_collection(all_poems)
            
            # Save results
            self.save_alt_lit_collection(all_poems)
        
        return all_poems
    
    def analyze_collection(self, poems):
        """Analyze the aesthetic characteristics of collected poems"""
        if not poems:
            return
        
        print(f"\n=== COLLECTION ANALYSIS ===")
        
        # Source breakdown
        sources = {}
        for poem in poems:
            source = poem['source']
            sources[source] = sources.get(source, 0) + 1
        
        print("Sources:")
        for source, count in sources.items():
            print(f"  {source}: {count} poems")
        
        # Style analysis
        total_lowercase_lines = sum(poem['style_indicators']['lowercase_lines'] for poem in poems)
        total_lines = sum(poem['style_indicators']['total_lines'] for poem in poems)
        
        if total_lines > 0:
            lowercase_ratio = total_lowercase_lines / total_lines
            print(f"Lowercase line ratio: {lowercase_ratio:.2%} (higher = more alt-lit style)")
        
        # Modern reference analysis
        modern_refs = sum(poem['style_indicators']['modern_references'] for poem in poems)
        print(f"Modern references found: {modern_refs}")
        
        # Show most alt-lit style poems
        alt_lit_scores = []
        for poem in poems:
            indicators = poem['style_indicators']
            score = (
                indicators['lowercase_lines'] * 2 +
                indicators['modern_references'] * 3 +
                indicators['casual_language'] * 1 +
                indicators['emotional_words'] * 1
            )
            alt_lit_scores.append((score, poem))
        
        alt_lit_scores.sort(reverse=True)
        
        print(f"\nMost alt-lit style poems:")
        for score, poem in alt_lit_scores[:3]:
            print(f"  Score {score}: {poem['title'][:40]}...")
    
    def save_alt_lit_collection(self, poems, filename="alt_lit_poetry_collection"):
        """Save collection with metadata"""
        
        # Full JSON with metadata
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(poems, f, indent=2, ensure_ascii=False)
        
        # Training format for neural networks
        with open(f"{filename}_training.txt", 'w', encoding='utf-8') as f:
            for poem in poems:
                f.write("<POEM_START>\n")
                f.write(poem['text'])
                f.write("\n<POEM_END>\n\n")
        
        # Readable format
        with open(f"{filename}_readable.txt", 'w', encoding='utf-8') as f:
            for poem in poems:
                f.write(f"=== {poem['title']} ===\n")
                f.write(f"Source: {poem['source']}\n")
                f.write(f"URL: {poem['url']}\n")
                f.write(f"Alt-lit indicators: {poem['style_indicators']}\n\n")
                f.write(poem['text'])
                f.write(f"\n\n{'='*60}\n\n")
        
        print(f"✓ Collection saved:")
        print(f"  - {filename}.json (full metadata)")
        print(f"  - {filename}_training.txt (neural network format)")
        print(f"  - {filename}_readable.txt (human readable)")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.driver.quit()
        except:
            pass

# Usage function
def run_alt_lit_scraper():
    """Run the alt-lit poetry scraper"""
    scraper = None
    try:
        scraper = AltLitPoetryScraper(headless=False)  # Visual mode for debugging
        poems = scraper.run_alt_lit_scraping_session()
        
        if poems:
            print(f"\n✓ Successfully collected {len(poems)} alt-lit style poems")
            print("These poems should match the Dream Boy Book Club aesthetic!")
            return poems
        else:
            print("✗ No poems collected. Check the debug output above.")
            return []
            
    except Exception as e:
        print(f"Scraper failed: {e}")
        return []
        
    finally:
        if scraper:
            scraper.cleanup()

if __name__ == "__main__":
    alt_lit_poems = run_alt_lit_scraper()
