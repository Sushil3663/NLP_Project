"""
Distant supervision utilities for creating reading comprehension training data from news passages.
Generates question-answer pairs using article titles and content.
"""

import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DistantSupervisionGenerator:
    """Generate training data using distant supervision from news articles."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Question templates for different types of information
        self.question_templates = {
            'what': [
                "What is {}?",
                "What does {} mean?",
                "What happened with {}?",
                "What is the main topic about {}?",
            ],
            'who': [
                "Who is {}?",
                "Who was involved in {}?",
                "Who mentioned {}?",
            ],
            'when': [
                "When did {} happen?",
                "When was {} mentioned?",
            ],
            'where': [
                "Where did {} take place?",
                "Where is {} located?",
            ],
            'why': [
                "Why is {} important?",
                "Why did {} happen?",
            ],
            'how': [
                "How does {} work?",
                "How is {} related to the topic?",
            ]
        }
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """Extract potential entities using simple heuristics."""
        entities = []
        
        # Find capitalized words/phrases (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(capitalized_pattern, text)
        
        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
        
        for match in matches:
            if match not in common_words and len(match) > 2:
                entities.append(match)
        
        # Find quoted text (potential important terms)
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)
        entities.extend(quoted_matches)
        
        return list(set(entities))  # Remove duplicates
    
    def find_answer_in_passage(self, passage: str, entity: str) -> Optional[Tuple[str, int, int]]:
        """Find answer span for entity in passage."""
        sentences = sent_tokenize(passage)
        
        for sentence in sentences:
            if entity.lower() in sentence.lower():
                # Try to extract a meaningful answer around the entity
                words = word_tokenize(sentence)
                entity_words = word_tokenize(entity.lower())
                
                # Find entity position in sentence
                for i in range(len(words) - len(entity_words) + 1):
                    if [w.lower() for w in words[i:i + len(entity_words)]] == entity_words:
                        # Extract context around entity (±5 words)
                        start_idx = max(0, i - 5)
                        end_idx = min(len(words), i + len(entity_words) + 5)
                        
                        answer_words = words[start_idx:end_idx]
                        answer_text = ' '.join(answer_words)
                        
                        # Find answer span in original passage
                        start_pos, end_pos = self.tokenizer.find_answer_span(passage, answer_text)
                        
                        if start_pos != -1 and end_pos != -1:
                            return answer_text, start_pos, end_pos
        
        return None
    
    def generate_questions_from_title(self, title: str, passage: str) -> List[Dict[str, Any]]:
        """Generate questions from article title."""
        questions = []
        
        # Use title as answer, generate questions
        title_clean = title.strip()
        if len(title_clean) < 10 or len(title_clean) > 200:
            return questions
        
        # Find title span in passage
        answer_span = self.find_answer_in_passage(passage, title_clean)
        if not answer_span:
            # Try with first sentence of passage as answer
            sentences = sent_tokenize(passage)
            if sentences:
                first_sentence = sentences[0]
                answer_span = self.find_answer_in_passage(passage, first_sentence)
                if answer_span:
                    title_clean = first_sentence
        
        if answer_span:
            answer_text, start_pos, end_pos = answer_span
            
            # Generate different types of questions
            question_types = ['what', 'who', 'when', 'where']
            
            for q_type in question_types:
                if q_type in self.question_templates:
                    template = random.choice(self.question_templates[q_type])
                    
                    # Create question based on type
                    if q_type == 'what':
                        question = "What is this article about?"
                    elif q_type == 'who':
                        question = "Who is mentioned in this article?"
                    elif q_type == 'when':
                        question = "When did this happen?"
                    elif q_type == 'where':
                        question = "Where did this take place?"
                    else:
                        continue
                    
                    questions.append({
                        'question': question,
                        'context': passage,  # Use 'context' for SQuAD compatibility
                        'answers': [{
                            'text': answer_text,
                            'answer_start': start_pos
                        }],
                        'question_type': q_type,
                        'source': 'title',
                        'is_impossible': False
                    })
        
        return questions
    
    def generate_questions_from_entities(self, passage: str) -> List[Dict[str, Any]]:
        """Generate questions from entities in passage."""
        questions = []
        
        # Extract entities
        entities = self.extract_entities_simple(passage)
        
        # Limit number of entities to avoid too many questions
        entities = entities[:5]
        
        for entity in entities:
            # Find answer span for entity
            answer_span = self.find_answer_in_passage(passage, entity)
            
            if answer_span:
                answer_text, start_pos, end_pos = answer_span
                
                # Generate questions about this entity
                question_types = ['what', 'who']
                
                for q_type in question_types:
                    if q_type in self.question_templates:
                        template = random.choice(self.question_templates[q_type])
                        question = template.format(entity)
                        
                        questions.append({
                            'question': question,
                            'context': passage,
                            'answers': [{
                                'text': answer_text,
                                'answer_start': start_pos
                            }],
                            'question_type': q_type,
                            'source': 'entity',
                            'entity': entity,
                            'is_impossible': False
                        })
        
        return questions
    
    def generate_questions_from_sentences(self, passage: str) -> List[Dict[str, Any]]:
        """Generate questions from key sentences in passage."""
        questions = []
        
        sentences = sent_tokenize(passage)
        
        # Focus on first few sentences (usually most important)
        key_sentences = sentences[:3]
        
        for sentence in key_sentences:
            if len(sentence) < 20 or len(sentence) > 300:
                continue
            
            # Use sentence as answer
            answer_span = self.find_answer_in_passage(passage, sentence)
            
            if answer_span:
                answer_text, start_pos, end_pos = answer_span
                
                # Generate generic questions
                generic_questions = [
                    "What does the article say?",
                    "What is mentioned in the text?",
                    "What information is provided?",
                ]
                
                question = random.choice(generic_questions)
                
                questions.append({
                    'question': question,
                    'context': passage,
                    'answers': [{
                        'text': answer_text,
                        'answer_start': start_pos
                    }],
                    'question_type': 'generic',
                    'source': 'sentence',
                    'is_impossible': False
                })
        
        return questions
    
    def create_training_data(self, passages: List[Dict[str, Any]], 
                           max_questions_per_passage: int = 3) -> List[Dict[str, Any]]:
        """Create training data from passages using distant supervision."""
        all_questions = []
        
        for passage_data in tqdm(passages, desc="Generating questions"):
            passage_text = passage_data.get('text', '')
            title = passage_data.get('title', '')
            
            if len(passage_text) < 100:  # Skip very short passages
                continue
            
            questions = []
            
            # Generate questions from title
            if title:
                title_questions = self.generate_questions_from_title(title, passage_text)
                questions.extend(title_questions)
            
            # Generate questions from entities
            entity_questions = self.generate_questions_from_entities(passage_text)
            questions.extend(entity_questions)
            
            # Generate questions from sentences
            sentence_questions = self.generate_questions_from_sentences(passage_text)
            questions.extend(sentence_questions)
            
            # Limit questions per passage
            if len(questions) > max_questions_per_passage:
                questions = random.sample(questions, max_questions_per_passage)
            
            # Add passage metadata to questions
            for question in questions:
                question.update({
                    'passage_id': passage_data.get('passage_id', ''),
                    'article_id': passage_data.get('article_id', ''),
                    'url': passage_data.get('url', ''),
                    'publisher': passage_data.get('publisher', ''),
                    'date': passage_data.get('date', ''),
                })
            
            all_questions.extend(questions)
        
        print(f"Generated {len(all_questions)} questions from {len(passages)} passages")
        return all_questions
    
    def create_impossible_questions(self, passages: List[Dict[str, Any]], 
                                  num_impossible: int = 1000) -> List[Dict[str, Any]]:
        """Create impossible questions (no answer in passage) for SQuAD 2.0 style training."""
        impossible_questions = []
        
        # Generic impossible questions
        impossible_templates = [
            "What is the exact date of the moon landing?",
            "Who invented the telephone?",
            "What is the capital of Mars?",
            "When was the internet created?",
            "How many people live in Antarctica?",
            "What is the speed of light?",
            "Who wrote Romeo and Juliet?",
            "What is the largest ocean on Earth?",
            "When did World War II end?",
            "What is the chemical symbol for gold?",
        ]
        
        for i in range(min(num_impossible, len(passages))):
            passage_data = passages[i]
            passage_text = passage_data.get('text', '')
            
            if len(passage_text) < 100:
                continue
            
            # Select random impossible question
            question = random.choice(impossible_templates)
            
            impossible_questions.append({
                'question': question,
                'context': passage_text,
                'answers': [],
                'is_impossible': True,
                'question_type': 'impossible',
                'source': 'impossible',
                'passage_id': passage_data.get('passage_id', ''),
                'article_id': passage_data.get('article_id', ''),
                'url': passage_data.get('url', ''),
                'publisher': passage_data.get('publisher', ''),
                'date': passage_data.get('date', ''),
            })
        
        print(f"Generated {len(impossible_questions)} impossible questions")
        return impossible_questions

def create_distant_supervision_data(passages: List[Dict[str, Any]], 
                                  tokenizer,
                                  output_file: str,
                                  max_questions_per_passage: int = 3,
                                  impossible_ratio: float = 0.2) -> List[Dict[str, Any]]:
    """Create complete distant supervision dataset."""
    
    generator = DistantSupervisionGenerator(tokenizer)
    
    # Generate answerable questions
    answerable_questions = generator.create_training_data(passages, max_questions_per_passage)
    
    # Generate impossible questions
    num_impossible = int(len(answerable_questions) * impossible_ratio)
    impossible_questions = generator.create_impossible_questions(passages, num_impossible)
    
    # Combine all questions
    all_questions = answerable_questions + impossible_questions
    
    # Shuffle
    random.shuffle(all_questions)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_questions)} questions to {output_file}")
    print(f"  Answerable: {len(answerable_questions)}")
    print(f"  Impossible: {len(impossible_questions)}")
    
    return all_questions

if __name__ == "__main__":
    # Test distant supervision generation
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from squad_data import SimpleTokenizer
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Test with sample passages
    sample_passages = [
        {
            'passage_id': 'test_1',
            'title': 'OpenAI Releases New AI Model',
            'text': 'OpenAI announced today the release of their new artificial intelligence model called GPT-4. The model shows significant improvements in reasoning and language understanding. The company said the model will be available through their API next month.',
            'url': 'https://example.com/test1',
            'publisher': 'Tech News'
        },
        {
            'passage_id': 'test_2',
            'title': 'Climate Change Impact on Agriculture',
            'text': 'Scientists at Stanford University published a study showing how climate change affects crop yields. The research indicates that rising temperatures could reduce wheat production by 15% over the next decade. The study was published in Nature Climate Change journal.',
            'url': 'https://example.com/test2',
            'publisher': 'Science Daily'
        }
    ]
    
    # Build simple vocabulary
    all_texts = []
    for passage in sample_passages:
        all_texts.append(passage['title'])
        all_texts.append(passage['text'])
    
    tokenizer.build_vocab(all_texts, max_vocab_size=1000)
    
    # Generate questions
    questions = create_distant_supervision_data(
        sample_passages, 
        tokenizer, 
        'test_distant_supervision.json',
        max_questions_per_passage=2
    )
    
    # Print sample questions
    print("\nSample generated questions:")
    for i, q in enumerate(questions[:5]):
        print(f"\n{i+1}. Question: {q['question']}")
        print(f"   Answer: {q['answer_text']}")
        print(f"   Span: ({q['start_position']}, {q['end_position']})")
        print(f"   Source: {q['source']}")
        print(f"   Impossible: {q.get('is_impossible', False)}")
    
    # Clean up test file
    if os.path.exists('test_distant_supervision.json'):
        os.remove('test_distant_supervision.json')