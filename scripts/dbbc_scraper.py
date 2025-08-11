#!/usr/bin/env python3
"""
Multi-poem DBBC Scraper
Handles pages with multiple poems (like ashley-d-escobar)
"""

import time
import json
import re
import requests
import random
from bs4 import BeautifulSoup
import sys
import os

class MultiPoemDBBCScraper:
    def __init__(self, debug=False):
        self.debug = debug
        self.poems_collected = []
        self.failed_urls = []
        self.skipped_urls = []
        self.session = self.setup_session()
        
    def setup_session(self):
        """Setup requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session
    
    def detect_content_type(self, soup):
        """Detect if page has poetry, prose, visual art, or other content"""
        # Check for high image count (likely photography/visual art)
        images = soup.find_all('img')
        if len(images) > 8:
            return 'visual_art'
        
        content_div = soup.find('div', class_='Content-outer')
        if not content_div:
            return 'unknown'
        
        full_text = content_div.get_text(separator='\n', strip=True)
        lines = full_text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if len(non_empty_lines) < 3:
            return 'minimal'
        
        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        
        # Prose detection
        if avg_line_length > 500:
            return 'prose'
        
        # Poetry indicators
        short_lines = [l for l in non_empty_lines if len(l) < 60]
        short_line_ratio = len(short_lines) / len(non_empty_lines)
        
        # Experimental poetry
        if len(non_empty_lines) < 10 and any(len(l) > 200 for l in non_empty_lines):
            return 'experimental_poetry'
        
        # Standard poetry
        if short_line_ratio > 0.5 or avg_line_length < 80:
            return 'poetry'
        
        # Mixed form
        if avg_line_length > 200:
            return 'narrative'
        
        return 'unknown'
    
    def extract_poems_from_page(self, url, response_text):
        """Extract one or more poems from DBBC page"""
        soup = BeautifulSoup(response_text, 'html.parser')
        
        # Get author name from URL
        author_slug = url.split('/')[-1]
        author_name = author_slug.replace('-', ' ').title()
        
        # Detect content type
        content_type = self.detect_content_type(soup)
        
        if self.debug:
            print(f"  Content type detected: {content_type}")
        
        # Skip non-poetry content
        if content_type in ['visual_art', 'minimal', 'unknown', 'narrative', 'prose']:
            if self.debug:
                print(f"  ✗ Skipping: {content_type} content")
            return []
        
        # Get content container
        content_div = soup.find('div', class_='Content-outer')
        if not content_div:
            return []
        
        full_text = content_div.get_text(separator='\n', strip=True)
        lines = full_text.split('\n')
        
        # Find bio start
        bio_start_idx = self.find_bio_start(lines, author_name)
        
        # Check for multiple poem markers
        poem_markers = self.find_poem_markers(lines, bio_start_idx)
        
        if len(poem_markers) > 1:
            # Multiple poems on page
            if self.debug:
                print(f"  Found {len(poem_markers)} potential poems on page")
            return self.extract_multiple_poems(lines, author_name, poem_markers, bio_start_idx, url, content_type)
        else:
            # Single poem
            poem = self.extract_single_poem(lines, author_name, bio_start_idx, url, content_type)
            return [poem] if poem else []
    
    def find_bio_start(self, lines, author_name):
        """Find where bio section starts"""
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Check for bio patterns with author name
            if any(name_part.lower() in line_lower for name_part in author_name.split()):
                bio_patterns = [
                    'is a literary', 'is an author', 'is a poet', 'is a writer',
                    'is an artist', 'selected her', 'selected his', 'graduated from',
                    'based in', 'lives in', 'residing in', '(b.', 'born'
                ]
                if any(pattern in line_lower for pattern in bio_patterns):
                    return i
        return len(lines)
    
    def find_poem_markers(self, lines, bio_start_idx):
        """Find potential poem title/start markers"""
        markers = []
        
        # Look for decorated titles or section breaks
        for i, line in enumerate(lines[:bio_start_idx]):
            line = line.strip()
            
            # Skip author name line
            if i == 1:
                continue
            
            # Decorated titles
            if line and (
                any(char in line for char in ['✿', '❀', '☁️', '🍒', '🎀', '♡', '~', '*', '҉', '【', '】']) or
                (len(line) < 50 and line.isupper()) or
                (i > 1 and lines[i-1].strip() == '' and lines[i+1].strip() == '')  # Isolated title
            ):
                markers.append(i)
            
            # Look for poem titles like "Night Hotel" or "Snuggle Bear"
            elif i > 10 and len(line) < 30 and line and not line[0].islower():
                # Potential poem title if preceded by empty line
                if i > 0 and not lines[i-1].strip():
                    markers.append(i)
        
        # If no markers found but we have content, assume it starts after author name
        if not markers and bio_start_idx > 2:
            markers.append(2)
        
        return markers
    
    def extract_multiple_poems(self, lines, author_name, poem_markers, bio_start_idx, url, content_type):
        """Extract multiple poems from a single page"""
        poems = []
        
        for i, marker_idx in enumerate(poem_markers):
            # Determine poem boundaries
            start_idx = marker_idx
            if i < len(poem_markers) - 1:
                end_idx = poem_markers[i + 1]
            else:
                end_idx = bio_start_idx
            
            # Extract poem
            poem_lines = lines[start_idx:end_idx]
            
            # Get title and body
            title = self.extract_title_from_lines(poem_lines)
            body = self.extract_body_from_lines(poem_lines, title)
            
            if body and len(body) > 50:
                if self.validate_poem_text(body):
                    poems.append({
                        'url': url,
                        'author': author_name,
                        'title': title,
                        'text': body,
                        'content_type': content_type,
                        'length': len(body),
                        'line_count': len(body.split('\n')),
                        'source': 'Dream Boy Book Club',
                        'dbbc_score': self.calculate_dbbc_score(body)
                    })
        
        return poems
    
    def extract_single_poem(self, lines, author_name, bio_start_idx, url, content_type):
        """Extract a single poem from page"""
        # Find poem start (after author name)
        poem_start_idx = -1
        for i, line in enumerate(lines[:min(10, bio_start_idx)]):
            if author_name.lower() in line.lower() and len(line) < 50:
                poem_start_idx = i + 1
                break
        
        if poem_start_idx == -1:
            poem_start_idx = 2
        
        # Extract poem lines
        poem_lines = lines[poem_start_idx:bio_start_idx]
        
        # Clean up
        while poem_lines and not poem_lines[0].strip():
            poem_lines.pop(0)
        while poem_lines and not poem_lines[-1].strip():
            poem_lines.pop()
        
        if not poem_lines:
            return None
        
        # Get title and body
        title = self.extract_title_from_lines(lines[:poem_start_idx + 5])
        body = '\n'.join(poem_lines)
        
        if content_type == 'experimental_poetry':
            # For experimental poetry, be more lenient
            if len(body) < 100:
                return None
        elif not self.validate_poem_text(body):
            return None
        
        return {
            'url': url,
            'author': author_name,
            'title': title,
            'text': body,
            'content_type': content_type,
            'length': len(body),
            'line_count': len(body.split('\n')),
            'source': 'Dream Boy Book Club',
            'dbbc_score': self.calculate_dbbc_score(body)
        }
    
    def extract_title_from_lines(self, lines):
        """Extract title from poem lines"""
        for line in lines[:5]:
            line = line.strip()
            if line and (
                any(char in line for char in ['✿', '❀', '☁️', '🍒', '🎀', '♡', '~', '*', '҉', '【', '】']) or
                (len(line) < 80 and line.isupper())
            ):
                # Clean up decoration
                title = re.sub(r'[*~˗ˏˋ꒰꒱ˊˎ]+', '', line).strip()
                return title if title else "Untitled"
        
        # Use first short line as title
        for line in lines[:3]:
            if 5 < len(line) < 50:
                return line.strip()
        
        return "Untitled"
    
    def extract_body_from_lines(self, lines, title):
        """Extract poem body, skipping title and author lines"""
        body_lines = []
        found_content_start = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip title line
            if title != "Untitled" and title in line:
                continue
            
            # Skip author name lines (common patterns)
            if not found_content_start:
                # Check if it's an author line
                if line_stripped and (
                    'Ashley' in line or 'Escobar' in line or
                    'Melissa' in line or 'Aliz' in line or
                    len(line_stripped.split()) <= 3 and line_stripped[0].isupper()
                ):
                    # Likely author name, skip
                    continue
                elif line_stripped:
                    # Found start of actual poem content
                    found_content_start = True
            
            # Skip decorative lines that are just the title
            if '☆' in line or '【' in line or '҉' in line:
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    continue
            
            body_lines.append(line)
        
        # Clean up
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        
        # Remove author name if it appears at the end
        if body_lines and len(body_lines[-1]) < 50:
            last_line = body_lines[-1].strip()
            if any(name in last_line for name in ['Ashley', 'Escobar', 'Melissa', 'Aliz']):
                body_lines.pop()
        
        return '\n'.join(body_lines)
    
    def validate_poem_text(self, text):
        """Validate that text is actually a poem"""
        if not text or len(text) < 50:
            return False
        
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        if len(non_empty_lines) < 2:
            return False
        
        # Check for UI elements
        ui_terms = ['Sign In', 'Cart', 'Menu', 'Search', 'Subscribe']
        if any(term in text for term in ui_terms):
            return False
        
        # Calculate characteristics
        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        
        # Too prose-like
        if avg_line_length > 200:
            return False
        
        # Check for poetic elements
        text_lower = text.lower()
        poetic_elements = 0
        
        if avg_line_length < 80:
            poetic_elements += 1
        if avg_line_length < 40:
            poetic_elements += 1
        if 'i ' in text_lower or "i'" in text_lower:
            poetic_elements += 1
        if any(word in text_lower for word in ['love', 'heart', 'feel', 'dream', 'fuck', 'cum', 'want', 'need']):
            poetic_elements += 1
        if any(word in text_lower for word in ['sun', 'moon', 'body', 'skin', 'mouth', 'eyes']):
            poetic_elements += 1
        
        return poetic_elements >= 2
    
    def calculate_dbbc_score(self, text):
        """Calculate DBBC aesthetic score"""
        score = 0
        text_lower = text.lower()
        
        # Contemporary references
        modern_terms = ['phone', 'internet', 'instagram', 'text', 'uber', 'netflix']
        score += sum(5 for term in modern_terms if term in text_lower)
        
        # Casual language
        casual_terms = ['like', 'just', 'really', 'kinda', 'babe']
        score += sum(3 for term in casual_terms if term in text_lower)
        
        # Vulnerability
        vulnerable_phrases = ['i feel', 'i think', 'i want', 'i need']
        score += sum(4 for phrase in vulnerable_phrases if phrase in text_lower)
        
        # Explicit content (DBBC characteristic)
        explicit_terms = ['fuck', 'cum', 'sex', 'slut']
        score += sum(6 for term in explicit_terms if term in text_lower)
        
        # Structure
        lines = text.split('\n')
        if 5 < len(lines) < 50:
            score += 5
        
        # Lowercase style
        lowercase_lines = sum(1 for line in lines if line and line[0].islower())
        if lowercase_lines > len(lines) * 0.3:
            score += 8
        
        return score
    
    def scrape_url(self, url):
        """Scrape a single URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            poems = self.extract_poems_from_page(url, response.text)
            
            if poems:
                for poem in poems:
                    self.poems_collected.append(poem)
                    print(f"  ✓ Extracted: \"{poem['title'][:30]}...\" ({poem['length']} chars, score: {poem['dbbc_score']})")
                return True
            else:
                # Check content type
                soup = BeautifulSoup(response.text, 'html.parser')
                content_type = self.detect_content_type(soup)
                if content_type in ['visual_art', 'narrative', 'prose']:
                    self.skipped_urls.append({'url': url, 'reason': content_type})
                    print(f"  ⊘ Skipped: {content_type} content")
                else:
                    self.failed_urls.append(url)
                    print(f"  ✗ No poems extracted")
                return False
                
        except Exception as e:
            self.failed_urls.append(url)
            print(f"  ✗ Error: {e}")
            return False
    
    def get_dbbc_urls(self):
        """Get comprehensive list of DBBC author URLs"""
        return [
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
            'https://www.dreamboybook.club/michael-washington',
            'https://www.dreamboybook.club/madlen-stafford',
            'https://www.dreamboybook.club/annika-gavlak',
            'https://www.dreamboybook.club/rae-wayland',
            'https://www.dreamboybook.club/poppy-cockburn',
            'https://www.dreamboybook.club/carly-jane-dagen',
            'https://www.dreamboybook.club/lucia-auerbach',
            'https://www.dreamboybook.club/ana-palacios',
            'https://www.dreamboybook.club/jane-dabate',
            'https://www.dreamboybook.club/david-san-miguel',
            'https://www.dreamboybook.club/meghan-gunn',
            'https://www.dreamboybook.club/nora-rose-tomas',
            'https://www.dreamboybook.club/greta-schledorn',
            'https://www.dreamboybook.club/sofia-hoefig',
            'https://www.dreamboybook.club/maria-kirsch',
            'https://www.dreamboybook.club/luna-sferdianu',
            'https://www.dreamboybook.club/lillian-mottern',
            'https://www.dreamboybook.club/paige-greco',
            'https://www.dreamboybook.club/ana-carrete',
            'https://www.dreamboybook.club/jaime-barash',
            'https://www.dreamboybook.club/sophie-knox-peters',
            'https://www.dreamboybook.club/matthew-tyler-vorce',
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
            'https://www.dreamboybook.club/allison-billmeyer',
            'https://www.dreamboybook.club/sophia-tempest',
            'https://www.dreamboybook.club/aurora-bodenhamer',
            'https://www.dreamboybook.club/lauren-milici',
            'https://www.dreamboybook.club/izzy-capulong',
            'https://www.dreamboybook.club/jade-wootton',
            'https://www.dreamboybook.club/sophie-gillet',
            'https://www.dreamboybook.club/jason-salvant',
            'https://www.dreamboybook.club/bambi-fields',
            'https://www.dreamboybook.club/luc-m',
            'https://www.dreamboybook.club/a-r-strain',
            'https://www.dreamboybook.club/adam-harb',
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
            'https://www.dreamboybook.club/katie-friedman',
            'https://www.dreamboybook.club/ulyses-razo',
            'https://www.dreamboybook.club/casper-kelly',
            'https://www.dreamboybook.club/emily-robinson',
            'https://www.dreamboybook.club/annie-lou-martin',
            'https://www.dreamboybook.club/alexandra-naughton',
            'https://www.dreamboybook.club/stephanie-yue-duhem',
            'https://www.dreamboybook.club/clarke-e-andros',
            'https://www.dreamboybook.club/nate-waggoner',
            'https://www.dreamboybook.club/lemmy-yaakova',
            'https://www.dreamboybook.club/emily-leibert',
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
            'https://www.dreamboybook.club/lana-valdez',
            'https://www.dreamboybook.club/nicholas-wilder-forman',
            'https://www.dreamboybook.club/matthew-ciazza',
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
            'https://www.dreamboybook.club/mila-rae-mancuso',
            'https://www.dreamboybook.club/alex-here',
            'https://www.dreamboybook.club/olivia-zarzycki',
            'https://www.dreamboybook.club/mariana-rodriguez',
            'https://www.dreamboybook.club/bronwen-lam',
            'https://www.dreamboybook.club/joyce-safdiah',
            'https://www.dreamboybook.club/swan-scissors',
            'https://www.dreamboybook.club/frank-demma',
            'https://www.dreamboybook.club/juliette-jeffers',
            'https://www.dreamboybook.club/josh-rodriguez',
            'https://www.dreamboybook.club/jessica-abughattas',
            'https://www.dreamboybook.club/marianne-agnes',
            'https://www.dreamboybook.club/katja-grober',
            'https://www.dreamboybook.club/joshua-bohnsack',
            'https://www.dreamboybook.club/nastasia-koulich',
            'https://www.dreamboybook.club/alisa-christiane-otte',
            'https://www.dreamboybook.club/ayla-mccarthy-combes'
        ]

    def scrape_all_dbbc_pages(self, limit=None):
        """Scrape all DBBC author pages with multi-poem handling"""
        urls = self.get_dbbc_urls()
        if limit:
            urls = urls[:limit]
        
        print(f"\n=== MULTI-POEM DBBC SCRAPER - {len(urls)} AUTHOR PAGES ===\n")
        
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Scraping: {url}")
            
            # Respectful delay
            if i > 1:
                delay = random.uniform(2, 4)
                time.sleep(delay)
            
            self.scrape_url(url)
        
        return self.poems_collected

    def test_problematic_urls(self):
        """Test on the problematic URLs"""
        test_urls = [
            'https://www.dreamboybook.club/ashley-d-escobar',
            'https://www.dreamboybook.club/melissa-aliz',
            'https://www.dreamboybook.club/valley-r-lee',
            'https://www.dreamboybook.club/taryn-segal',
        ]
        
        print("TESTING PROBLEMATIC URLS")
        print("="*50)
        
        for url in test_urls:
            print(f"\nTesting: {url}")
            self.scrape_url(url)
            time.sleep(2)
        
        # Summary
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Poems collected: {len(self.poems_collected)}")
        print(f"Failed: {len(self.failed_urls)}")
        print(f"Skipped: {len(self.skipped_urls)}")
        
        if self.poems_collected:
            print("\nCollected poems:")
            for poem in self.poems_collected:
                print(f"  - {poem['author']}: \"{poem['title'][:30]}...\" ({poem['line_count']} lines)")
        
        return self.poems_collected
    
    def save_results(self, filename_prefix="multi_poem_dbbc_collection"):
        """Save scraped poems to files"""
        if not self.poems_collected:
            print("No poems to save!")
            return
        
        # Sort by DBBC score
        self.poems_collected.sort(key=lambda p: p['dbbc_score'], reverse=True)
        
        # Save as JSON
        json_file = f"{filename_prefix}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.poems_collected, f, indent=2, ensure_ascii=False)
        
        # Save training format
        training_file = f"{filename_prefix}_training.txt"
        with open(training_file, 'w', encoding='utf-8') as f:
            for poem in self.poems_collected:
                f.write("<POEM_START>\n")
                f.write(poem['text'])
                f.write("\n<POEM_END>\n\n")
        
        # Save readable format
        readable_file = f"{filename_prefix}_readable.txt"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write("=== MULTI-POEM DBBC POETRY COLLECTION ===\n")
            f.write(f"Total poems: {len(self.poems_collected)}\n")
            f.write(f"Failed URLs: {len(self.failed_urls)}\n")
            f.write(f"Skipped URLs: {len(self.skipped_urls)}\n")
            f.write(f"Success rate: {len(self.poems_collected) / (len(self.poems_collected) + len(self.failed_urls)) * 100:.1f}%\n\n")
            
            for i, poem in enumerate(self.poems_collected, 1):
                f.write(f"#{i} - {poem['title']}\n")
                f.write(f"Author: {poem['author']}\n")
                f.write(f"DBBC Score: {poem['dbbc_score']}\n")
                f.write(f"URL: {poem['url']}\n")
                f.write(f"Length: {poem['length']} chars, {poem['line_count']} lines\n\n")
                f.write(poem['text'])
                f.write(f"\n\n{'='*60}\n\n")
        
        print(f"\n✓ Collection saved to:")
        print(f"  - {json_file}")
        print(f"  - {training_file}")
        print(f"  - {readable_file}")
    
    def print_summary(self):
        """Print scraping summary"""
        total_attempts = len(self.poems_collected) + len(self.failed_urls)
        success_rate = (len(self.poems_collected) / total_attempts * 100) if total_attempts > 0 else 0
        
        print("\n" + "="*50)
        print("SCRAPING SUMMARY")
        print("="*50)
        print(f"Total URLs attempted: {total_attempts}")
        print(f"Successful extractions: {len(self.poems_collected)}")
        print(f"Failed extractions: {len(self.failed_urls)}")
        print(f"Skipped URLs: {len(self.skipped_urls)}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.poems_collected:
            scores = [p['dbbc_score'] for p in self.poems_collected]
            print(f"\nDBBC Aesthetic Scores:")
            print(f"  Average: {sum(scores) / len(scores):.1f}")
            print(f"  Min: {min(scores)}")
            print(f"  Max: {max(scores)}")
            
            total_chars = sum(p['length'] for p in self.poems_collected)
            print(f"\nContent collected:")
            print(f"  Total characters: {total_chars:,}")
            print(f"  Average poem length: {total_chars // len(self.poems_collected):,} chars")
            
            print(f"\nTop 3 poems by DBBC score:")
            for i, poem in enumerate(self.poems_collected[:3], 1):
                print(f"  {i}. \"{poem['title'][:40]}...\" (score: {poem['dbbc_score']})")
        
        if self.failed_urls and self.debug:
            print(f"\nFailed URLs:")
            for url in self.failed_urls[:10]:
                print(f"  - {url}")
            if len(self.failed_urls) > 10:
                print(f"  ... and {len(self.failed_urls) - 10} more")


def main():
    """Run multi-poem DBBC scraper"""
    print("🚀 MULTI-POEM DBBC SCRAPER")
    print("Comprehensive scraping with multi-poem extraction capability\n")
    
    # Parse command line arguments
    debug = '--debug' in sys.argv
    test_only = '--test' in sys.argv
    limit = None
    for arg in sys.argv:
        if arg.startswith('--limit='):
            limit = int(arg.split('=')[1])
    
    scraper = MultiPoemDBBCScraper(debug=debug)
    
    if test_only:
        print("Running test mode on problematic URLs...")
        poems = scraper.test_problematic_urls()
        filename_prefix = "multi_poem_test"
    else:
        print("Running full website scrape...")
        poems = scraper.scrape_all_dbbc_pages(limit=limit)
        filename_prefix = "multi_poem_dbbc_collection"
    
    # Print summary
    scraper.print_summary()
    
    # Save results
    if poems:
        scraper.save_results(filename_prefix)
        print("\n✅ Scraping complete! Check the output files for your poetry collection.")
    else:
        print("\n❌ No poems were collected. Check the debug output for issues.")
    
    return len(poems)


if __name__ == "__main__":
    exit(0 if main() > 0 else 1)
