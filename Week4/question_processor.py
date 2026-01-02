"""
Question preprocessing utilities for the inference pipeline.
Handles question analysis, entity extraction, and type classification.
"""

import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

class QuestionProcessor:
    """Processes and analyzes questions for the QA pipeline."""
    
    def __init__(self):
        self.question_types = {
            'what': ['what', 'which'],
            'who': ['who', 'whom'],
            'when': ['when'],
            'where': ['where'],
            'why': ['why'],
            'how': ['how'],
            'yes_no': ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should']
        }
        
        # Date patterns for temporal questions
        self.date_patterns = [
            r'\b\d{4}\b',  # Year (e.g., 2024)
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(today|yesterday|tomorrow|last week|next week|last month|next month|last year|next year)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_date_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns]
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of question based on question words."""
        question_lower = question.lower().strip()
        
        # Remove punctuation for analysis
        question_words = word_tokenize(question_lower)
        
        if not question_words:
            return 'unknown'
        
        first_word = question_words[0]
        
        # Check each question type
        for q_type, keywords in self.question_types.items():
            if first_word in keywords:
                return q_type
        
        # Check for implicit yes/no questions
        if any(word in question_words[:3] for word in self.question_types['yes_no']):
            return 'yes_no'
        
        return 'unknown'
    
    def extract_entities(self, question: str) -> List[Dict[str, Any]]:
        """Extract named entities from the question."""
        entities = []
        
        if nlp is None:
            # Fallback: simple capitalized word extraction
            capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
            for word in capitalized_words:
                entities.append({
                    'text': word,
                    'label': 'UNKNOWN',
                    'start': question.find(word),
                    'end': question.find(word) + len(word)
                })
        else:
            # Use spaCy for entity extraction
            doc = nlp(question)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
                })
        
        return entities
    
    def extract_temporal_hints(self, question: str) -> List[Dict[str, Any]]:
        """Extract temporal information from the question."""
        temporal_hints = []
        
        for pattern in self.compiled_date_patterns:
            matches = pattern.finditer(question)
            for match in matches:
                temporal_hints.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'date_mention'
                })
        
        # Look for relative time expressions
        relative_time_patterns = [
            (r'\b(recent|recently|latest|new|current|now)\b', 'recent'),
            (r'\b(old|older|past|previous|former|earlier)\b', 'past'),
            (r'\b(future|upcoming|next|coming|will)\b', 'future'),
            (r'\b(today|this week|this month|this year)\b', 'current_period'),
            (r'\b(last \w+|previous \w+|\w+ ago)\b', 'past_period'),
            (r'\b(next \w+|coming \w+|in \d+ \w+)\b', 'future_period')
        ]
        
        for pattern, hint_type in relative_time_patterns:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                temporal_hints.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': hint_type
                })
        
        return temporal_hints
    
    def analyze_question_focus(self, question: str) -> Dict[str, Any]:
        """Analyze what the question is focusing on."""
        question_lower = question.lower()
        
        focus_indicators = {
            'definition': ['what is', 'what are', 'define', 'definition of', 'meaning of'],
            'explanation': ['how does', 'how do', 'explain', 'why does', 'why do'],
            'identification': ['who is', 'who are', 'which is', 'which are'],
            'location': ['where is', 'where are', 'where did', 'where does'],
            'time': ['when is', 'when are', 'when did', 'when does', 'when will'],
            'quantity': ['how many', 'how much', 'how long', 'how often'],
            'comparison': ['difference between', 'compare', 'versus', 'vs', 'better than'],
            'causation': ['why', 'because', 'reason', 'cause', 'result']
        }
        
        detected_focus = []
        for focus_type, indicators in focus_indicators.items():
            for indicator in indicators:
                if indicator in question_lower:
                    detected_focus.append(focus_type)
                    break
        
        return {
            'primary_focus': detected_focus[0] if detected_focus else 'general',
            'all_focus_types': detected_focus
        }
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question."""
        # Remove question words and common stop words
        stop_words = {
            'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whom',
            'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could',
            'will', 'would', 'should', 'the', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about'
        }
        
        # Tokenize and filter
        words = word_tokenize(question.lower())
        keywords = []
        
        for word in words:
            # Keep words that are:
            # - Not stop words
            # - Longer than 2 characters
            # - Alphanumeric
            if (word not in stop_words and 
                len(word) > 2 and 
                word.isalnum()):
                keywords.append(word)
        
        return keywords
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process a question and return comprehensive analysis."""
        # Clean the question
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        
        # Perform all analyses
        analysis = {
            'original_question': question,
            'question_type': self.classify_question_type(question),
            'entities': self.extract_entities(question),
            'temporal_hints': self.extract_temporal_hints(question),
            'focus_analysis': self.analyze_question_focus(question),
            'keywords': self.extract_keywords(question),
            'processed_at': datetime.now().isoformat()
        }
        
        # Add derived features
        analysis['has_entities'] = len(analysis['entities']) > 0
        analysis['has_temporal_hints'] = len(analysis['temporal_hints']) > 0
        analysis['is_temporal_question'] = (
            analysis['question_type'] == 'when' or 
            analysis['has_temporal_hints'] or
            'time' in analysis['focus_analysis']['all_focus_types']
        )
        
        # Generate search hints
        analysis['search_hints'] = self.generate_search_hints(analysis)
        
        return analysis
    
    def generate_search_hints(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hints for improving search/retrieval."""
        hints = {
            'boost_recent': False,
            'boost_entities': [],
            'preferred_sources': [],
            'date_range': None,
            'query_expansion': []
        }
        
        # Boost recent content for temporal questions
        if analysis['is_temporal_question']:
            temporal_hints = analysis['temporal_hints']
            recent_indicators = ['recent', 'latest', 'new', 'current', 'today', 'this week', 'this month']
            
            if any(hint['type'] == 'recent' or hint['text'].lower() in recent_indicators 
                   for hint in temporal_hints):
                hints['boost_recent'] = True
        
        # Extract important entities for boosting
        for entity in analysis['entities']:
            if entity['label'] in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                hints['boost_entities'].append(entity['text'])
        
        # Query expansion based on question type
        question_type = analysis['question_type']
        keywords = analysis['keywords']
        
        if question_type == 'what' and 'definition' in analysis['focus_analysis']['all_focus_types']:
            hints['query_expansion'].extend(['definition', 'meaning', 'explanation'])
        elif question_type == 'how':
            hints['query_expansion'].extend(['process', 'method', 'way', 'steps'])
        elif question_type == 'why':
            hints['query_expansion'].extend(['reason', 'cause', 'because', 'due to'])
        
        return hints

def preprocess_question(question: str) -> Dict[str, Any]:
    """Convenience function to preprocess a single question."""
    processor = QuestionProcessor()
    return processor.process_question(question)

if __name__ == "__main__":
    # Test the question processor
    processor = QuestionProcessor()
    
    test_questions = [
        "What is artificial intelligence?",
        "Who invented the telephone in 1876?",
        "When did World War II end?",
        "Where is the Eiffel Tower located?",
        "Why do birds migrate?",
        "How does photosynthesis work?",
        "Is climate change real?",
        "What are the latest developments in quantum computing?",
        "Who is the current president of the United States?",
        "When was the last time there was a solar eclipse?"
    ]
    
    print("Question Processing Test Results:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        analysis = processor.process_question(question)
        
        print(f"   Type: {analysis['question_type']}")
        print(f"   Focus: {analysis['focus_analysis']['primary_focus']}")
        print(f"   Keywords: {', '.join(analysis['keywords'][:5])}")
        
        if analysis['entities']:
            entities_str = ', '.join([f"{e['text']} ({e['label']})" for e in analysis['entities'][:3]])
            print(f"   Entities: {entities_str}")
        
        if analysis['temporal_hints']:
            temporal_str = ', '.join([f"{t['text']} ({t['type']})" for t in analysis['temporal_hints'][:2]])
            print(f"   Temporal: {temporal_str}")
        
        if analysis['search_hints']['boost_recent']:
            print(f"   Hint: Boost recent content")
        
        if analysis['search_hints']['boost_entities']:
            print(f"   Hint: Boost entities: {', '.join(analysis['search_hints']['boost_entities'][:2])}")
    
    print("\n" + "=" * 50)