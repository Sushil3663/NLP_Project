"""
Passage chunking module for splitting articles into retrievable segments.
Creates 200-400 token passages with 50-token overlap for optimal retrieval.
"""

import re
from typing import List, Dict, Any, Tuple
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PassageChunker:
    """Handles intelligent chunking of articles into passages."""
    
    def __init__(self, min_tokens: int = 200, max_tokens: int = 400, overlap_tokens: int = 50):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using NLTK word tokenizer."""
        return len(word_tokenize(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving quality."""
        sentences = sent_tokenize(text)
        
        # Filter and clean sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and self.count_tokens(sent) > 2:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def create_passages(self, sentences: List[str]) -> List[str]:
        """Create passages from sentences with proper overlap."""
        if not sentences:
            return []
        
        passages = []
        current_passage = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed max_tokens, finalize current passage
            if current_tokens + sentence_tokens > self.max_tokens and current_passage:
                passage_text = ' '.join(current_passage)
                if self.count_tokens(passage_text) >= self.min_tokens:
                    passages.append(passage_text)
                
                # Create overlap for next passage
                overlap_passage = []
                overlap_tokens = 0
                
                # Add sentences from the end of current passage for overlap
                for j in range(len(current_passage) - 1, -1, -1):
                    sent_tokens = self.count_tokens(current_passage[j])
                    if overlap_tokens + sent_tokens <= self.overlap_tokens:
                        overlap_passage.insert(0, current_passage[j])
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                current_passage = overlap_passage
                current_tokens = overlap_tokens
            
            # Add current sentence
            current_passage.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Add final passage if it meets minimum requirements
        if current_passage:
            passage_text = ' '.join(current_passage)
            if self.count_tokens(passage_text) >= self.min_tokens:
                passages.append(passage_text)
        
        return passages
    
    def chunk_article(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single article into passages with metadata."""
        if not article.get('text'):
            return []
        
        # Split into sentences
        sentences = self.split_into_sentences(article['text'])
        
        if not sentences:
            return []
        
        # Create passages
        passages = self.create_passages(sentences)
        
        # Create passage objects with metadata
        passage_objects = []
        for i, passage_text in enumerate(passages):
            passage_obj = {
                'text': passage_text,
                'article_id': article.get('url', f"article_{id(article)}"),
                'passage_id': f"{article.get('url', f'article_{id(article)}')}_{i}",
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'date': article.get('date', ''),
                'publisher': article.get('publisher', ''),
                'author': article.get('author', ''),
                'summary': article.get('summary', ''),
                'passage_index': i,
                'total_passages': len(passages),
                'token_count': self.count_tokens(passage_text)
            }
            passage_objects.append(passage_obj)
        
        return passage_objects
    
    def chunk_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple articles into passages."""
        all_passages = []
        
        for article in articles:
            passages = self.chunk_article(article)
            all_passages.extend(passages)
        
        return all_passages

def create_training_queries(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create training queries from article titles and first sentences."""
    queries = []
    
    for passage in passages:
        # Use article title as query
        if passage.get('title') and len(passage['title'].strip()) > 10:
            query = {
                'query': passage['title'],
                'positive_passage_id': passage['passage_id'],
                'article_id': passage['article_id'],
                'type': 'title'
            }
            queries.append(query)
        
        # Use first sentence as query (for first passage only)
        if passage.get('passage_index', 0) == 0:
            sentences = sent_tokenize(passage['text'])
            if sentences and len(sentences[0]) > 20:
                query = {
                    'query': sentences[0],
                    'positive_passage_id': passage['passage_id'],
                    'article_id': passage['article_id'],
                    'type': 'first_sentence'
                }
                queries.append(query)
    
    return queries

if __name__ == "__main__":
    # Test the chunker
    sample_article = {
        'title': 'Test Article About AI',
        'text': '''This is the first sentence of the article. It introduces the topic of artificial intelligence.
        
        The second paragraph discusses machine learning in detail. It explains how neural networks work and their applications.
        
        The third paragraph covers deep learning. It mentions convolutional neural networks and their use in computer vision.
        
        The final paragraph concludes the article. It summarizes the key points about AI and machine learning.''',
        'url': 'https://example.com/test',
        'date': '2024-01-01',
        'publisher': 'Test Publisher'
    }
    
    chunker = PassageChunker(min_tokens=50, max_tokens=100, overlap_tokens=20)
    passages = chunker.chunk_article(sample_article)
    
    print(f"Created {len(passages)} passages:")
    for i, passage in enumerate(passages):
        print(f"\nPassage {i+1} ({passage['token_count']} tokens):")
        print(passage['text'][:100] + "...")