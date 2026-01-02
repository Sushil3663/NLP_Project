"""
Text utilities for cleaning and normalizing news articles.
Handles Unicode normalization, boilerplate removal, and text preprocessing.
"""

import re
import unicodedata
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextCleaner:
    """Handles text cleaning and normalization for news articles."""
    
    def __init__(self):
        # Common boilerplate patterns to remove
        self.boilerplate_patterns = [
            r'This story originally appeared on.*',
            r'Updated \d+/\d+/\d+ at.*',
            r'Subscribe to WIRED.*',
            r'Let us know what you think.*',
            r'Submit a letter to the editor.*',
            r'Special offer for.*readers.*',
            r'Don\'t miss future.*',
            r'Additional reporting by.*',
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.boilerplate_patterns]
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters and fix encoding issues."""
        if not text:
            return ""
        
        # Normalize Unicode to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Fix common encoding issues
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '—': '-', '–': '-',  # Em/en dashes
            '…': '...',          # Ellipsis
            '\xa0': ' ',         # Non-breaking space
            '\u200b': '',        # Zero-width space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text from news articles."""
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """Apply comprehensive text cleaning."""
        if not text:
            return ""
        
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Remove boilerplate
        text = self.remove_boilerplate(text)
        
        # Fix paragraph boundaries
        text = self.fix_paragraph_boundaries(text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def fix_paragraph_boundaries(self, text: str) -> str:
        """Preserve and fix paragraph boundaries."""
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Clean each paragraph
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # Filter very short paragraphs
                # Ensure sentences end properly
                if not para.endswith(('.', '!', '?', '"', "'")):
                    para += '.'
                cleaned_paragraphs.append(para)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text while preserving quality."""
        sentences = sent_tokenize(text)
        
        # Filter sentences
        filtered_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # Keep sentences with reasonable length and content
            if (len(sent) > 10 and 
                len(word_tokenize(sent)) > 3 and
                not sent.isupper() and  # Skip all-caps sentences
                not re.match(r'^[^a-zA-Z]*$', sent)):  # Skip non-alphabetic
                filtered_sentences.append(sent)
        
        return filtered_sentences

def clean_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single article and return processed version."""
    cleaner = TextCleaner()
    
    cleaned_article = article.copy()
    
    # Clean main text
    if 'text' in article and article['text']:
        cleaned_article['text'] = cleaner.clean_text(article['text'])
    
    # Clean title
    if 'title' in article and article['title']:
        cleaned_article['title'] = cleaner.normalize_unicode(article['title']).strip()
    
    # Clean summary
    if 'summary' in article and article['summary']:
        cleaned_article['summary'] = cleaner.clean_text(article['summary'])
    
    return cleaned_article

if __name__ == "__main__":
    # Test the text cleaner
    sample_text = """
    This is a test article with "smart quotes" and—em dashes.
    It has multiple    spaces and weird\xa0characters.
    
    This story originally appeared on Ars Technica.
    Updated 4/2/2024, 1:23 pm ET to include additional details.
    """
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(sample_text)
    print("Original:")
    print(repr(sample_text))
    print("\nCleaned:")
    print(repr(cleaned))