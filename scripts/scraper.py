import time
import json
import re
import requests
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class ExpandedContemporaryPoetryScraper:
    def __init__(self, headless=True, debug=False):
        self.poems_found = []
        self.debug = debug
        self.setup_stealth_firefox(headless)
        self.setup_session()
        
    def setup_session(self):
        """Setup requests session with rotation"""
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        })
    
    def setup_stealth_firefox(self, headless=True):
        """Setup Firefox with stealth"""
        try:
            firefox_options = FirefoxOptions()
            if headless:
                firefox_options.add_argument("--headless")
            
            # Stealth preferences
            firefox_options.set_preference("general.useragent.override", random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0'
            ]))
            
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            firefox_options.set_preference("permissions.default.image", 2)  # Disable images for speed
            
            self.driver = webdriver.Firefox(options=firefox_options)
            self.driver.implicitly_wait(15)
            self.wait = WebDriverWait(self.driver, 30)
            
            print("✓ Stealth Firefox setup successful")
            
        except Exception as e:
            print(f"✗ Firefox setup failed: {e}")
            raise
    
    def random_delay(self, min_delay=0.8, max_delay=2.3):
        """Human-like delays"""
        time.sleep(random.uniform(min_delay, max_delay))

    def extract_hyperlinks(self,url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad status codes
        except requests.RequestException as e:
            print(f"Error fetching the page: {e}")
        return []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = []

        for tag in soup.find_all('a', href=True):
            full_url = urljoin(url, tag['href'])  # Convert relative URLs to absolute
            links.append(full_url)

        return links
    
    def get_expanded_poetry_sources(self):
        """Comprehensive list of contemporary poetry sources"""
        
        sources = [
            # Community Poetry Sites (known to work)
            {
                'name': 'dream boy book club',
                'urls': [
                        'https://www.dreamboybook.club/abby-romine',
                        'https://www.dreamboybook.club/sahaj-kaur',
'https://www.dreamboybook.club/ashley-d-escobar',
'https://www.dreamboybook.club/natalie-gilda',
'https://www.dreamboybook.club/caleb-f-stocco',
'https://www.dreamboybook.club/nestan-nikouradze',
'https://www.dreamboybook.club/emma-newman-holden',
'https://www.dreamboybook.club/stella-parker',
'https://www.dreamboybook.club/john-ling',
'https://www.dreamboybook.club/lydia-mckimm',
'https://www.dreamboybook.club/grace-helen',
'https://www.dreamboybook.club/alex-goodale',
'https://dreamboybook.club/michael-washington',
'https://www.dreamboybook.club/madlen-stafford',
'https://www.dreamboybook.club/annika-gavlak',
'https://www.dreamboybook.club/rae-wayland',
'https://www.dreamboybook.club/poppy-cockburn',
'https://www.dreamboybook.club/carly-jane-dagen',
'https://www.dreamboybook.club/lucia-auerbach',
'https://www.dreamboybook.club/ana-palacios',
'https://dreamboybook.club/jane-dabate',
'https://dreamboybook.club/david-san-miguel',
'https://dreamboybook.club/meghan-gunn',
'https://dreamboybook.club/nora-rose-tomas',
'https://dreamboybook.club/greta-schledorn',
'https://dreamboybook.club/sofia-hoefig',
'https://dreamboybook.club/maria-kirsch',
'https://dreamboybook.club/luna-sferdianu',
'https://dreamboybook.club/lillian-mottern',
'https://dreamboybook.club/paige-greco',
'https://dreamboybook.club/ana-carrete',
'https://dreamboybook.club/jaime-barash',
'https://dreamboybook.club/sophie-knox-peters',
'https://dreamboybook.club/matthew-tyler-vorce',
'https://www.dreamboybook.club/luce-childs',
'https://www.dreamboybook.club/taryn-segal',
'https://www.dreamboybook.club/valley-r-lee',
'https://www.dreamboybook.club/melissa-aliz',
'https://www.dreamboybook.club/eva-cat-kuhn',
'https://www.dreamboybook.club/madison-nash',
'https://www.dreamboybook.club/kd-sims',
'https://www.dreamboybook.club/kaia-polanska-richardson',
'https://www.dreamboybook.club/laura-mota-juang',
'https://www.dreamboybook.club/ashliene-mcmenamy',
'https://www.dreamboybook.club/ophelia-arc',
'https://dreamboybook.club/allison-billmeyer',
'https://www.dreamboybook.club/stella-parker',
'https://www.dreamboybook.club/sophia-tempest',
'https://dreamboybook.club/aurora-bodenhamer',
'https://dreamboybook.club/lauren-milici',
'https://www.dreamboybook.club/lauren-milici',
'https://www.dreamboybook.club/izzy-capulong',
'https://www.dreamboybook.club/jade-wootton',
'https://www.dreamboybook.club/sophie-gillet',
'https://www.dreamboybook.club/jason-salvant',
'https://www.dreamboybook.club/bambi-fields',
'https://dreamboybook.club/luc-m',
'https://www.dreamboybook.club/a-r-strain',
'https://www.dreamboybook.club/adam-harb',
'https://www.dreamboybook.club/alex-goodale',
'https://www.dreamboybook.club/melanie-robinson',
'https://www.dreamboybook.club/samantha-sewell',
'https://www.dreamboybook.club/emily-cox',
'https://www.dreamboybook.club/danielle-altman',
'https://www.dreamboybook.club/jack-ludkey',
'https://www.dreamboybook.club/reilly-tuesday',
'https://www.dreamboybook.club/maanasa',
'https://www.dreamboybook.club/filip-fufezan',
'https://www.dreamboybook.club/kate-durbin',
'https://www.dreamboybook.club/charlotte-loesch',
'https://www.dreamboybook.club/rebecca-hochman-fisher',
'https://www.dreamboybook.club/kat-thanopoulos',
'https://www.dreamboybook.club/lilly-hogan',
'https://www.dreamboybook.club/chayton-pabich-danyla',
'https://www.dreamboybook.club/naomi-leigh',
'https://www.dreamboybook.club/liam-serwin',
'https://www.dreamboybook.club/fiona-flynn',
'https://www.dreamboybook.club/sean-avery-medlin',
'https://www.dreamboybook.club/farah-abouzeid',
'https://www.dreamboybook.club/olivia-kamer',
'https://www.dreamboybook.club/nina-potischman',
'https://dreamboybook.club/katie-friedman',
'https://dreamboybook.club/ulyses-razo',
'https://dreamboybook.club/casper-kelly',
'https://dreamboybook.club/emily-robinson',
'https://www.dreamboybook.club/annie-lou-martin',
'https://www.dreamboybook.club/alexandra-naughton',
'https://www.dreamboybook.club/stephanie-yue-duhem',
'https://www.dreamboybook.club/sofia-hoefig',
'https://www.dreamboybook.club/clarke-e-andros',
'https://www.dreamboybook.club/nate-waggoner',
'https://www.dreamboybook.club/lemmy-yaakova',
'https://www.dreamboybook.club/nestan-nikouradze',
'https://www.dreamboybook.club/emily-leibert',
'https://www.dreamboybook.club/filip-fufezan',
'https://www.dreamboybook.club/maya-osep',
'https://www.dreamboybook.club/claire-benedicta-mclaughlin',
'https://www.dreamboybook.club/ruby-elliott-zuckerman',
'https://www.dreamboybook.club/caroline-ouellette',
'https://www.dreamboybook.club/heart-white',
'https://www.dreamboybook.club/peyton-gatewood',
'https://www.dreamboybook.club/lindsey-goodrow',
'https://www.dreamboybook.club/lotte-latham',
'https://www.dreamboybook.club/chelsea-becker',
'https://www.dreamboybook.club/jerusha-crone',
'https://www.dreamboybook.club/parker-love-bowling',
'https://www.dreamboybook.club/sarah-elda',
'https://www.dreamboybook.club/claudia-elena-rodriguez',
'https://www.dreamboybook.club/isabelle-joy-stephen',
'https://www.dreamboybook.club/kate-nerone',
'https://www.dreamboybook.club/rae-wayland',
'https://www.dreamboybook.club/lana-valdez',
'https://www.dreamboybook.club/nicholas-wilder-forman',
'https://www.dreamboybook.club/annika-gavlak',
'https://www.dreamboybook.club/matthew-ciazza',
'https://www.dreamboybook.club/ashley-d-escobar',
'https://www.dreamboybook.club/carmen-vega',
'https://www.dreamboybook.club/cora-lee',
'https://www.dreamboybook.club/sophia-howells',
'https://www.dreamboybook.club/rowan-bennetti',
'https://www.dreamboybook.club/ember-knight',
'https://www.dreamboybook.club/ashla-c-r',
'https://www.dreamboybook.club/cullen-arbaugh',
'https://www.dreamboybook.club/calla-selicious',
'https://www.dreamboybook.club/nicola-maye-goldberg',
'https://www.dreamboybook.club/benin-gardner',
'https://www.dreamboybook.club/maria-kirsch',
'https://www.dreamboybook.club/mila-rae-mancuso',
'https://www.dreamboybook.club/alex-here',
'https://www.dreamboybook.club/kaia-polanska-richardson',
'https://www.dreamboybook.club/olivia-zarzycki',
'https://www.dreamboybook.club/mariana-rodriguez',
'https://www.dreamboybook.club/bronwen-lam',
'https://www.dreamboybook.club/rebecca-hochman-fisher',
'https://www.dreamboybook.club/joyce-safdiah',
'https://www.dreamboybook.club/swan-scissors',
'https://www.dreamboybook.club/fiona-flynn',
'https://www.dreamboybook.club/frank-demma',
'https://www.dreamboybook.club/juliette-jeffers',
'https://www.dreamboybook.club/josh-rodriguez',
'https://www.dreamboybook.club/jessica-abughattas',
'https://www.dreamboybook.club/marianne-agnes',
'https://www.dreamboybook.club/katja-grober',
'https://www.dreamboybook.club/joshua-bohnsack',
'https://www.dreamboybook.club/nastasia-koulich',
'https://www.dreamboybook.club/alisa-christiane-otte',
'https://www.dreamboybook.club/ayla-mccarthy-combes',
                ],
                    
                'type': 'canon'
            },
            {
                'name': 'All Poetry',
                'urls': [
                    'https://allpoetry.com/poems/best',
                    'https://allpoetry.com/contemporary-poems',
                    'https://allpoetry.com/poems/modern'
                ],
                'type': 'community'
            },
            {
                'name': 'Hello Poetry',
                'urls': [
                    'https://hellopoetry.com/poems/recent/',
                    'https://hellopoetry.com/poetry/contemporary/',
                    'https://hellopoetry.com/poetry/love/',
                    'https://hellopoetry.com/poetry/sad/'
                ],
                'type': 'community'
            },
            {
                'name': 'Poem Hunter',
                'urls': [
                    'https://www.poemhunter.com/poems/contemporary/',
                    'https://www.poemhunter.com/poem/modern/',
                    'https://www.poemhunter.com/poems/love/',
                    'https://www.poemhunter.com/poems/life/'
                ],
                'type': 'community'
            },
            
            # Literary Magazines & Journals
            {
                'name': 'Poetry Magazine Online',
                'urls': [
                    'https://www.poetrymagazine.org/poems',
                    'https://www.poetrymagazine.org/browse?combine=&field_poem_themes_target_id=All&field_form_target_id=All&sort_by=created&sort_order=DESC'
                ],
                'type': 'magazine'
            },
            {
                'name': 'Literary Hub Poetry',
                'urls': [
                    'https://lithub.com/poetry/',
                    'https://lithub.com/tag/poems/'
                ],
                'type': 'magazine'
            },
            {
                'name': 'The Paris Review',
                'urls': [
                    'https://www.theparisreview.org/poetry',
                    'https://www.theparisreview.org/browse/poetry'
                ],
                'type': 'magazine'
            },
            {
                'name': 'Poetry Daily',
                'urls': [
                    'https://poems.com/',
                    'https://poems.com/archive.php'
                ],
                'type': 'daily'
            },
            
            # Contemporary/Alt-Lit Focused
            {
                'name': 'Button Poetry',
                'urls': [
                    'https://buttonpoetry.com/category/poems/',
                    'https://buttonpoetry.com/category/spoken-word/'
                ],
                'type': 'contemporary'
            },
            {
                'name': 'Split This Rock',
                'urls': [
                    'https://www.splitthisrock.org/poetry-database',
                    'https://www.splitthisrock.org/poems'
                ],
                'type': 'contemporary'
            },
            
            # University & Student Publications
            {
                'name': 'Adroit Journal',
                'urls': [
                    'https://www.theadroitjournal.org/issue-archive/',
                    'https://www.theadroitjournal.org/category/poetry/'
                ],
                'type': 'student'
            },
            {
                'name': 'Polyphony HS',
                'urls': [
                    'https://www.polyphonyhs.com/poetry',
                    'https://www.polyphonyhs.com/archive'
                ],
                'type': 'student'
            },
            
            # Open Access & Archives
            {
                'name': 'Poetry and Translation Centre',
                'urls': [
                    'https://poetryandtranslation.org/category/poetry/',
                    'https://poetryandtranslation.org/category/new-work/'
                ],
                'type': 'archive'
            },
            {
                'name': 'Verse Daily',
                'urls': [
                    'https://www.versedaily.org/',
                    'https://www.versedaily.org/archives.shtml'
                ],
                'type': 'daily'
            },
            
            # Emerging/Digital-First Publications
            {
                'name': 'Diode Poetry Journal',
                'urls': [
                    'https://www.diodepoetry.com/archive/',
                    'https://www.diodepoetry.com/current-issue/'
                ],
                'type': 'digital'
            },
            {
                'name': 'BOAAT Journal',
                'urls': [
                    'https://www.boaatpress.com/boaat-journal',
                    'https://www.boaatpress.com/archive'
                ],
                'type': 'digital'
            },
            {
                'name': 'Yes Poetry',
                'urls': [
                    'https://yespoetry.com/archive',
                    'https://yespoetry.com/category/poetry'
                ],
                'type': 'digital'
            },
            
            # Millennial/Gen-Z Focused
            {
                'name': 'Miracle Monocle',
                'urls': [
                    'https://miraclemonocle.com/category/poetry/',
                    'https://miraclemonocle.com/archives/'
                ],
                'type': 'millennial'
            },
            {
                'name': 'Glass Poetry Press',
                'urls': [
                    'https://www.glasspoetrypress.com/blog',
                    'https://www.glasspoetrypress.com/featured-poems'
                ],
                'type': 'millennial'
            },
            
            # Poetry Blogs & Personal Sites
            {
                'name': 'Poetry Society America',
                'urls': [
                    'https://poetrysociety.org/poems',
                    'https://poetrysociety.org/poetry-in-motion'
                ],
                'type': 'organization'
            },
            {
                'name': 'Poets.org Contemporary',
                'urls': [
                    'https://poets.org/poems?field_poem_themes_target_id=All&field_occasion_target_id=All&field_form_target_id=All&combine=contemporary',
                    'https://poets.org/poems?combine=modern'
                ],
                'type': 'organization'
            }
        ]
        
        return sources
    
    def scrape_all_expanded_sources(self):
        """Scrape from all expanded sources"""
        
        print("=== EXPANDED CONTEMPORARY POETRY SCRAPING ===")
        print("Targeting sources that match Dream Boy Book Club aesthetic...\n")
        
        sources = self.get_expanded_poetry_sources()
        all_poems = []
        
        for source in sources:
            print(f"\n=== SCRAPING {source['name']} ({source['type'].upper()}) ===")
            
            source_poems = []
            
            for url in source['urls']:
                try:
                    print(f"Loading: {url}")
                    
                    # Smart page loading
                    poems_from_page = self.scrape_page_intelligently(url, source['name'])
                    
                    if poems_from_page:
                        source_poems.extend(poems_from_page)
                        print(f"  ✓ Found {len(poems_from_page)} poems")
                    else:
                        print(f"  ✗ No poems found")
                    
                    self.random_delay(1, 2.6)
                    
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    continue
            
            if source_poems:
                all_poems.extend(source_poems)
                print(f"✓ Total from {source['name']}: {len(source_poems)} poems")
            else:
                print(f"✗ No poems collected from {source['name']}")
        
        return all_poems
    
    def scrape_page_intelligently(self, url, source_name):
        """Intelligent page scraping with multiple strategies"""
        
        try:
            # Load page with smart waiting
            self.driver.get(url)
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Human-like scrolling to trigger content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            self.random_delay(1, 1.5)
            self.driver.execute_script("window.scrollTo(0, 0);")
            self.random_delay(1, 2)
            
            # Extract content using multiple methods
            poems = []
            
            # Method 1: Look for poem-specific elements
            poems.extend(self.extract_poems_method_1())
            
            # Method 2: Look for article/post content
            poems.extend(self.extract_poems_method_2())
            
            # Method 3: General text extraction
            poems.extend(self.extract_poems_method_3())
            
            # Clean and deduplicate
            unique_poems = self.deduplicate_poems(poems)
            
            # Add source info
            for poem in unique_poems:
                poem['source'] = source_name
                poem['url'] = url
            
            return unique_poems
            
        except Exception as e:
            print(f"    Page scraping error: {e}")
            return []
    
    def extract_poems_method_1(self):
        """Method 1: Poetry-specific selectors"""
        poems = []
        
        poem_selectors = [
            '.poem', '.poetry', '.verse', '[class*="poem"]',
            'pre', 'blockquote', '.content pre', '.entry-content pre',
            '.poem-text', '.poetry-content', '.verse-content'
        ]
        
        for selector in poem_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                for element in elements:
                    text = element.text.strip()
                    if self.looks_like_contemporary_poetry(text):
                        poems.append({
                            'text': text,
                            'title': self.extract_title_from_element(element),
                            'method': f'poetry_selector_{selector}'
                        })
            except:
                continue
        
        return poems
    
    def extract_poems_method_2(self):
        """Method 2: Article/post content"""
        poems = []
        
        content_selectors = [
            'article', '.article', '.post', '.entry',
            '.content', '.main-content', '.entry-content',
            '.post-content', '.article-content'
        ]
        
        for selector in content_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                for element in elements:
                    # Look for poetry within the content
                    text_blocks = self.find_poetry_in_content(element)
                    
                    for text in text_blocks:
                        if self.looks_like_contemporary_poetry(text):
                            poems.append({
                                'text': text,
                                'title': self.extract_title_from_element(element),
                                'method': f'content_selector_{selector}'
                            })
            except:
                continue
        
        return poems
    
    def extract_poems_method_3(self):
        """Method 3: General text extraction"""
        poems = []
        
        try:
            # JavaScript extraction of substantial text blocks
            text_blocks = self.driver.execute_script("""
                var blocks = [];
                var elements = document.querySelectorAll('p, div, section');
                
                for (var i = 0; i < elements.length; i++) {
                    var text = elements[i].innerText || elements[i].textContent || '';
                    if (text.length > 50 && text.length < 2000) {
                        var lines = text.split('\\n').filter(line => line.trim().length > 0);
                        if (lines.length > 2 && lines.length < 50) {
                            blocks.push(text.trim());
                        }
                    }
                }
                
                return blocks;
            """)
            
            for text in text_blocks:
                if self.looks_like_contemporary_poetry(text):
                    poems.append({
                        'text': text,
                        'title': self.extract_title_from_text(text),
                        'method': 'javascript_extraction'
                    })
        except:
            pass
        
        return poems
    
    def find_poetry_in_content(self, element):
        """Find poetry within larger content blocks"""
        full_text = element.text.strip()
        
        # Split by common separators and look for poem-like sections
        potential_poems = []
        
        # Split by double line breaks (stanza separation)
        sections = re.split(r'\n\s*\n', full_text)
        
        for section in sections:
            section = section.strip()
            if 50 < len(section) < 1500:  # Reasonable poem length
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                
                # Check if section has poetry characteristics
                if len(lines) > 2:
                    avg_line_length = sum(len(line) for line in lines) / len(lines)
                    if 10 < avg_line_length < 100:  # Poetry-like line lengths
                        potential_poems.append(section)
        
        return potential_poems
    
    def extract_title_from_element(self, element):
        """Extract title from element context"""
        try:
            # Look for title in nearby elements
            parent = element.find_element(By.XPATH, '..')
            
            # Check for title/heading elements
            title_selectors = ['h1', 'h2', 'h3', 'h4', '.title', '.poem-title', '.entry-title']
            
            for selector in title_selectors:
                try:
                    title_elem = parent.find_element(By.CSS_SELECTOR, selector)
                    title = title_elem.text.strip()
                    if 3 < len(title) < 100:
                        return title
                except:
                    continue
            
            # Fallback: use first line of poem
            return self.extract_title_from_text(element.text)
            
        except:
            return "Untitled"
    
    def extract_title_from_text(self, text):
        """Extract title from poem text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if lines:
            first_line = lines[0]
            if len(first_line) < 60:
                return first_line
        
        # Use first few words
        words = text.split()[:6]
        return ' '.join(words) + ('...' if len(words) == 6 else '')
    
    def looks_like_contemporary_poetry(self, text):
        """Enhanced contemporary poetry detection"""
        if len(text) < 30:
            return False
        
        # Skip obvious metadata
        if self.is_website_metadata(text):
            return False
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
        
        # Contemporary poetry characteristics
        characteristics = [
            # Line structure
            10 < sum(len(line) for line in lines) / len(lines) < 80,
            
            # Personal voice
            text.lower().count('i ') > 0,
            
            # Modern/casual language
            any(word in text.lower() for word in [
                'like', 'just', 'really', 'feel', 'think', 'maybe', 'kinda'
            ]),
            
            # Emotional content
            any(word in text.lower() for word in [
                'love', 'heart', 'sad', 'happy', 'lonely', 'dream', 'remember', 'hurt', 'pain'
            ]),
            
            # Not prose-like
            not any(len(line) > 150 for line in lines),
            
            # Has line breaks
            len(lines) > 3,
            
            # Alt-lit indicators
            any(word in text.lower() for word in [
                'anxiety', 'depression', 'phone', 'text', 'internet', 'instagram'
            ]),
            
            # Poetic devices
            text.count('like a') > 0 or text.count('as if') > 0
        ]
        
        return sum(characteristics) >= 4
    
    def is_website_metadata(self, text):
        """Detect website metadata"""
        metadata_indicators = [
            'follow', 'founded', 'website', 'homepage', 'navigation',
            'subscribe', 'newsletter', 'rate this poem', 'copyright',
            'allpoetry.com', 'poemhunter.com', 'poetry.com',
            'window.datalayer', 'google analytics', 'advertisement'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in metadata_indicators)
    
    def deduplicate_poems(self, poems):
        """Remove duplicate poems"""
        unique_poems = []
        seen_signatures = set()
        
        for poem in poems:
            # Create signature for duplicate detection
            signature = poem['text'][:50] + str(len(poem['text']))
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_poems.append(poem)
        
        return unique_poems
    
    def calculate_alt_lit_score(self, text):
        """Calculate alt-lit aesthetic score"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        indicators = {
            'lowercase_lines': len([line for line in lines if line and line[0].islower()]),
            'first_person': text.lower().count('i '),
            'modern_references': sum(1 for word in [
                'phone', 'internet', 'app', 'instagram', 'twitter', 'text', 'wifi', 'laptop'
            ] if word in text.lower()),
            'emotional_words': sum(1 for word in [
                'anxiety', 'depression', 'lonely', 'sad', 'empty', 'broken', 'hurt', 'pain'
            ] if word in text.lower()),
            'casual_language': sum(1 for word in [
                'like', 'just', 'really', 'kinda', 'maybe', 'feel', 'think'
            ] if word in text.lower()),
            'vulnerability_markers': sum(1 for phrase in [
                'i feel', 'i think', 'i wish', 'i want', 'i need', 'i miss'
            ] if phrase in text.lower())
        }
        
        score = (
            indicators['lowercase_lines'] * 2 +
            indicators['modern_references'] * 4 +
            indicators['casual_language'] * 2 +
            indicators['emotional_words'] * 3 +
            indicators['vulnerability_markers'] * 4 +
            indicators['first_person'] * 1
        )
        
        # Bonus for good structure
        if 15 < sum(len(line) for line in lines) / len(lines) < 60:
            score += 8
        
        if text.count('\n\n') > 0:  # Stanza breaks
            score += 5
        
        return score, indicators
    
    def run_expanded_scraping(self):
        """Run the expanded scraping session"""
        
        print("🚀 EXPANDED CONTEMPORARY POETRY SCRAPER")
        print("Targeting Dream Boy Book Club aesthetic across multiple sources...\n")
        
        start_time = time.time()
        
        # Scrape all sources
        all_poems = self.scrape_all_expanded_sources()
        
        if all_poems:
            # Calculate alt-lit scores for all poems
            for poem in all_poems:
                score, indicators = self.calculate_alt_lit_score(poem['text'])
                poem['alt_lit_score'] = score
                poem['style_indicators'] = indicators
                poem['length'] = len(poem['text'])
                poem['line_count'] = len(poem['text'].split('\n'))
            
            # Sort by alt-lit score
            all_poems.sort(key=lambda p: p['alt_lit_score'], reverse=True)
            
            # Filter for decent scores
            quality_poems = [p for p in all_poems if p['alt_lit_score'] > 5]
            
            print(f"\n=== EXPANDED SCRAPING COMPLETE ===")
            print(f"Total items collected: {len(all_poems)}")
            print(f"Quality poems (score > 5): {len(quality_poems)}")
            print(f"Time taken: {(time.time() - start_time) / 60:.1f} minutes")
            
            if quality_poems:
                self.analyze_expanded_collection(quality_poems)
                self.save_expanded_collection(quality_poems)
                return quality_poems
            
        print("No quality poems collected from expanded sources.")
        return []
    
    def analyze_expanded_collection(self, poems):
        """Analyze the expanded collection"""
        
        print(f"\n=== EXPANDED COLLECTION ANALYSIS ===")
        
        # Source distribution
        sources = {}
        for poem in poems:
            source = poem['source']
            sources[source] = sources.get(source, 0) + 1
        
        print(f"Sources ({len(sources)} total):")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} poems")
        
        # Score distribution
        scores = [p['alt_lit_score'] for p in poems]
        print(f"\nScore distribution:")
        print(f"  Average: {sum(scores) / len(scores):.1f}")
        print(f"  Range: {min(scores)} - {max(scores)}")
        print(f"  Very high scoring (>30): {len([s for s in scores if s > 30])}")
        print(f"  High scoring (>15): {len([s for s in scores if s > 15])}")
        
        # Content analysis
        total_chars = sum(p['length'] for p in poems)
        print(f"\nContent:")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Average poem length: {total_chars // len(poems):,} characters")
        
        # Show top 5 alt-lit poems
        print(f"\nTop 5 alt-lit style poems:")
        for i, poem in enumerate(poems[:5]):
            print(f"{i+1}. \"{poem['title'][:40]}...\" (score: {poem['alt_lit_score']})")
            print(f"   Source: {poem['source']}")
            print(f"   Preview: {poem['text'][:80]}...")
    
    def save_expanded_collection(self, poems, filename="expanded_contemporary_poetry"):
        """Save the expanded collection"""
        
        # JSON with full metadata
        with open(f"/home/tgfm/workflows/autoencoder/dataset_poetry/{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(poems, f, indent=2, ensure_ascii=False)
        
        # Training format
        with open(f"/home/tgfm/workflows/autoencoder/dataset_poetry/{filename}_training.txt", 'w', encoding='utf-8') as f:
            for poem in poems:
                f.write("<POEM_START>\n")
                f.write(poem['text'])
                f.write("\n<POEM_END>\n\n")
        
        # Readable format
        with open(f"/home/tgfm/workflows/autoencoder/dataset_poetry/{filename}_readable.txt", 'w', encoding='utf-8') as f:
            f.write(f"=== EXPANDED CONTEMPORARY POETRY COLLECTION ===\n")
            f.write(f"Total poems: {len(poems)}\n")
            f.write("Sorted by alt-lit score\n\n")
            
            for i, poem in enumerate(poems):
                f.write(f"#{i+1} - {poem['title']}\n")
                f.write(f"Source: {poem['source']}\n")
                f.write(f"Alt-lit Score: {poem['alt_lit_score']}\n")
                f.write(f"Method: {poem.get('method', 'unknown')}\n")
                f.write(f"URL: {poem.get('url', '')}\n\n")
                f.write(poem['text'])
                f.write(f"\n\n{'='*60}\n\n")
        
        print(f"\n✓ Expanded collection saved as:")
        print(f"  - {filename}.json")
        print(f"  - {filename}_training.txt")
        print(f"  - {filename}_readable.txt")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.driver.quit()
        except:
            pass

def run_expanded_scraping(headless=True, debug=False):
    """Run the expanded scraping session"""
    
    scraper = None
    try:
        print("🎯 EXPANDED CONTEMPORARY POETRY SCRAPER")
        print("Targeting multiple sources for Dream Boy Book Club aesthetic")
        print("This will take longer but should collect much more data!\n")
        
        scraper = ExpandedContemporaryPoetryScraper(headless=headless, debug=debug)
        poems = scraper.run_expanded_scraping()
        
        if poems:
            print(f"\n🎉 EXPANDED SCRAPING SUCCESS!")
            print(f"Collected {len(poems)} contemporary poems")
            print("This should provide a much richer dataset for your autoencoder!")
            
            # Show sample of highest-scoring poem
            if poems:
                top_poem = poems[0]
                print(f"\nHighest-scoring poem:")
                print(f"Title: {top_poem['title']}")
                print(f"Source: {top_poem['source']}")
                print(f"Alt-lit Score: {top_poem['alt_lit_score']}")
                print(f"Preview: {top_poem['text'][:150]}...")
            
            return poems
        else:
            print("\n😞 Expanded scraping didn't collect any quality poems")
            return []
            
    except KeyboardInterrupt:
        print("\n⏸️  Scraping interrupted by user")
        return []
        
    except Exception as e:
        print(f"\n💥 Expanded scraper failed: {e}")
        return []
        
    finally:
        if scraper:
            scraper.cleanup()

if __name__ == "__main__":
    # Run expanded scraping
    poems = run_expanded_scraping(headless=False, debug=False)
