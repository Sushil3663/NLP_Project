"""
Answer aggregation and confidence scoring utilities.
Combines candidate answers from multiple passages and computes confidence scores.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from difflib import SequenceMatcher

class AnswerAggregator:
    """Aggregates and scores candidate answers from multiple passages."""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        
        # Answer type patterns for validation
        self.answer_patterns = {
            'person': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'organization': r'\b[A-Z][A-Z\s&.]+\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}\b|\b[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\b',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            'location': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        }
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer text for comparison."""
        if not answer:
            return ""
        
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Remove common articles and prepositions at the beginning
        answer = re.sub(r'^(the|a|an|in|on|at|to|for|of|with|by)\s+', '', answer)
        
        # Remove punctuation at the end
        answer = re.sub(r'[.!?;,]+$', '', answer)
        
        return answer
    
    def compute_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Compute similarity between two answers."""
        norm1 = self.normalize_answer(answer1)
        norm2 = self.normalize_answer(answer2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Substring match
        if norm1 in norm2 or norm2 in norm1:
            return 0.8
        
        # Sequence similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 and words2:
            word_overlap = len(words1 & words2) / len(words1 | words2)
            similarity = max(similarity, word_overlap)
        
        return similarity
    
    def cluster_similar_answers(self, candidates: List[Dict[str, Any]], 
                              similarity_threshold: float = 0.7) -> List[List[Dict[str, Any]]]:
        """Cluster similar candidate answers together."""
        if not candidates:
            return []
        
        clusters = []
        used = set()
        
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            
            cluster = [candidate]
            used.add(i)
            
            for j, other_candidate in enumerate(candidates[i + 1:], i + 1):
                if j in used:
                    continue
                
                similarity = self.compute_answer_similarity(
                    candidate['answer_text'], 
                    other_candidate['answer_text']
                )
                
                if similarity >= similarity_threshold:
                    cluster.append(other_candidate)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def score_answer_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score a cluster of similar answers."""
        if not cluster:
            return None
        
        # Choose representative answer (longest or most common)
        answer_texts = [c['answer_text'] for c in cluster]
        answer_counts = Counter(answer_texts)
        
        # Get most common answer, or longest if tie
        most_common = answer_counts.most_common(1)[0][0]
        representative = max(cluster, key=lambda x: len(x['answer_text']) 
                           if x['answer_text'] == most_common else 0)
        
        # Aggregate confidence scores
        confidences = [c['confidence'] for c in cluster]
        passage_scores = [c.get('passage_score', 0.5) for c in cluster]
        
        # Compute aggregate confidence
        # Higher confidence if multiple passages agree
        base_confidence = max(confidences)
        agreement_boost = min(0.3, (len(cluster) - 1) * 0.1)  # Boost for agreement
        avg_passage_score = np.mean(passage_scores)
        
        final_confidence = min(1.0, base_confidence + agreement_boost) * avg_passage_score
        
        # Collect supporting passages
        supporting_passages = []
        for candidate in cluster:
            passage_info = {
                'title': candidate.get('title', ''),
                'url': candidate.get('url', ''),
                'publisher': candidate.get('publisher', ''),
                'date': candidate.get('date', ''),
                'passage_score': candidate.get('passage_score', 0.0),
                'answer_confidence': candidate['confidence']
            }
            supporting_passages.append(passage_info)
        
        return {
            'answer_text': representative['answer_text'],
            'confidence': final_confidence,
            'support_count': len(cluster),
            'supporting_passages': supporting_passages,
            'answer_type': self.classify_answer_type(representative['answer_text']),
            'start_position': representative.get('start_position', -1),
            'end_position': representative.get('end_position', -1),
            'source_passage': representative.get('passage_text', '')
        }
    
    def classify_answer_type(self, answer: str) -> str:
        """Classify the type of answer based on patterns."""
        if not answer:
            return 'unknown'
        
        for answer_type, pattern in self.answer_patterns.items():
            if re.search(pattern, answer):
                return answer_type
        
        # Check for yes/no answers
        if answer.lower().strip() in ['yes', 'no', 'true', 'false']:
            return 'yes_no'
        
        # Check for short phrases (likely entities)
        if len(answer.split()) <= 3 and any(c.isupper() for c in answer):
            return 'entity'
        
        # Longer answers are likely explanations
        if len(answer.split()) > 10:
            return 'explanation'
        
        return 'phrase'
    
    def validate_answer_quality(self, answer: Dict[str, Any], question_analysis: Dict[str, Any]) -> float:
        """Validate answer quality based on question type and content."""
        if not answer or not answer.get('answer_text'):
            return 0.0
        
        answer_text = answer['answer_text'].strip()
        question_type = question_analysis.get('question_type', 'unknown')
        
        quality_score = 1.0
        
        # Check answer length appropriateness
        answer_length = len(answer_text.split())
        
        if question_type in ['who', 'what', 'where'] and answer_length > 20:
            quality_score *= 0.8  # Penalize very long answers for specific questions
        elif question_type in ['how', 'why'] and answer_length < 3:
            quality_score *= 0.7  # Penalize very short answers for explanatory questions
        
        # Check for empty or meaningless answers
        meaningless_patterns = [
            r'^(the|a|an|this|that|these|those)\s*$',
            r'^(yes|no)\s*$',  # Unless it's a yes/no question
            r'^\s*$',
            r'^[.!?]+$'
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, answer_text.lower()):
                if not (question_type == 'yes_no' and re.match(r'^(yes|no)\s*$', answer_text.lower())):
                    quality_score *= 0.3
        
        # Boost answers that contain question keywords
        question_keywords = question_analysis.get('keywords', [])
        answer_words = set(answer_text.lower().split())
        
        keyword_overlap = len(set(question_keywords) & answer_words)
        if keyword_overlap > 0:
            quality_score *= (1.0 + keyword_overlap * 0.1)
        
        # Check for entity consistency
        question_entities = [e['text'].lower() for e in question_analysis.get('entities', [])]
        if question_entities:
            answer_lower = answer_text.lower()
            entity_mentions = sum(1 for entity in question_entities if entity in answer_lower)
            if entity_mentions > 0:
                quality_score *= 1.2
        
        return min(1.0, quality_score)
    
    def aggregate_answers(self, 
                         candidates: List[Dict[str, Any]], 
                         question_analysis: Dict[str, Any],
                         max_answers: int = 3) -> List[Dict[str, Any]]:
        """
        Aggregate candidate answers and return top-scored answers.
        
        Args:
            candidates: List of candidate answers with confidence scores
            question_analysis: Analysis of the input question
            max_answers: Maximum number of answers to return
        
        Returns:
            List of aggregated answers sorted by confidence
        """
        if not candidates:
            return []
        
        # Filter out very low confidence candidates
        min_confidence = 0.1
        filtered_candidates = [c for c in candidates if c.get('confidence', 0) >= min_confidence]
        
        if not filtered_candidates:
            return []
        
        # Cluster similar answers
        clusters = self.cluster_similar_answers(filtered_candidates)
        
        # Score each cluster
        scored_answers = []
        for cluster in clusters:
            cluster_score = self.score_answer_cluster(cluster)
            if cluster_score:
                # Validate answer quality
                quality_score = self.validate_answer_quality(cluster_score, question_analysis)
                cluster_score['confidence'] *= quality_score
                cluster_score['quality_score'] = quality_score
                
                scored_answers.append(cluster_score)
        
        # Sort by confidence
        scored_answers.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return top answers
        return scored_answers[:max_answers]
    
    def should_return_answer(self, best_answer: Optional[Dict[str, Any]]) -> bool:
        """Determine if the best answer meets the confidence threshold."""
        if not best_answer:
            return False
        
        return best_answer.get('confidence', 0) >= self.confidence_threshold
    
    def create_fallback_response(self, 
                                top_passages: List[Dict[str, Any]], 
                                question: str) -> Dict[str, Any]:
        """Create fallback response when no confident answer is found."""
        return {
            'answer_text': 'not found',
            'confidence': 0.0,
            'is_fallback': True,
            'message': f'I could not find a confident answer to: "{question}"',
            'top_sources': [
                {
                    'title': p.get('title', 'Unknown'),
                    'url': p.get('url', ''),
                    'publisher': p.get('publisher', 'Unknown'),
                    'date': p.get('date', ''),
                    'relevance_score': p.get('rerank_score', p.get('score', 0.0))
                }
                for p in top_passages[:3]
            ]
        }

if __name__ == "__main__":
    # Test answer aggregation
    aggregator = AnswerAggregator(confidence_threshold=0.3)
    
    # Sample candidate answers
    candidates = [
        {
            'answer_text': 'artificial intelligence',
            'confidence': 0.8,
            'passage_score': 0.9,
            'title': 'AI Overview',
            'url': 'http://example.com/ai1',
            'publisher': 'Tech News'
        },
        {
            'answer_text': 'Artificial Intelligence',
            'confidence': 0.7,
            'passage_score': 0.8,
            'title': 'What is AI?',
            'url': 'http://example.com/ai2',
            'publisher': 'Science Daily'
        },
        {
            'answer_text': 'machine learning',
            'confidence': 0.6,
            'passage_score': 0.7,
            'title': 'ML Basics',
            'url': 'http://example.com/ml',
            'publisher': 'AI Weekly'
        },
        {
            'answer_text': 'AI technology',
            'confidence': 0.5,
            'passage_score': 0.6,
            'title': 'AI Tech',
            'url': 'http://example.com/ai3',
            'publisher': 'Tech Review'
        }
    ]
    
    # Sample question analysis
    question_analysis = {
        'question_type': 'what',
        'keywords': ['artificial', 'intelligence'],
        'entities': [],
        'focus_analysis': {'primary_focus': 'definition'}
    }
    
    # Aggregate answers
    aggregated = aggregator.aggregate_answers(candidates, question_analysis)
    
    print("Answer Aggregation Test Results:")
    print("=" * 40)
    
    for i, answer in enumerate(aggregated, 1):
        print(f"\n{i}. Answer: {answer['answer_text']}")
        print(f"   Confidence: {answer['confidence']:.3f}")
        print(f"   Support Count: {answer['support_count']}")
        print(f"   Answer Type: {answer['answer_type']}")
        print(f"   Quality Score: {answer['quality_score']:.3f}")
        print(f"   Supporting Passages: {len(answer['supporting_passages'])}")
    
    # Test confidence threshold
    best_answer = aggregated[0] if aggregated else None
    should_return = aggregator.should_return_answer(best_answer)
    
    print(f"\nShould return answer: {should_return}")
    print(f"Threshold: {aggregator.confidence_threshold}")
    
    if not should_return:
        fallback = aggregator.create_fallback_response(
            [{'title': 'Sample Article', 'url': 'http://example.com', 'publisher': 'Test'}],
            "What is artificial intelligence?"
        )
        print(f"Fallback response: {fallback['message']}")
    
    print("\n" + "=" * 40)