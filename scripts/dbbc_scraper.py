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


def main():
    """Run multi-poem scraper test"""
    print("🚀 MULTI-POEM DBBC SCRAPER TEST")
    print("Testing extraction of multiple poems per page\n")
    
    scraper = MultiPoemDBBCScraper(debug='--debug' in sys.argv)
    poems = scraper.test_problematic_urls()
    os.chdir('..')
    if poems:
        # Save test results
        with open('dataset_poetry/multi_poem_test.json', 'w') as f:
            json.dump(poems, f, indent=2, ensure_ascii=False)
        
        with open('dataset_poetry/multi_poem_test.txt', 'w') as f:
            for i, poem in enumerate(poems, 1):
                f.write(f"#{i} - {poem['title']}\n")
                f.write(f"Author: {poem['author']}\n")
                f.write(f"Score: {poem['dbbc_score']}\n\n")
                f.write(poem['text'])
                f.write("\n\n" + "="*60 + "\n\n")
        
        print(f"\n✅ Test complete! Saved {len(poems)} poems to multi_poem_test.json/txt")
    else:
        print("\n❌ No poems collected in test")
    
    return len(poems)


if __name__ == "__main__":
    exit(0 if main() > 0 else 1)
