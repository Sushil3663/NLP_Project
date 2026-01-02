"""
Deduplication module using MinHash for detecting near-identical passages.
Removes duplicate content while preserving passage diversity.
"""

import hashlib
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import random

class MinHashDeduplicator:
    """MinHash-based deduplication for text passages."""
    
    def __init__(self, num_hashes: int = 128, shingle_size: int = 3, threshold: float = 0.8):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[callable]:
        """Generate hash functions for MinHash."""
        hash_functions = []
        random.seed(42)  # For reproducibility
        
        for i in range(self.num_hashes):
            # Create unique hash function using different seeds
            seed = random.randint(1, 2**32 - 1)
            hash_functions.append(lambda x, s=seed: hash((x, s)) % (2**32 - 1))
        
        return hash_functions
    
    def create_shingles(self, text: str) -> Set[str]:
        """Create character-level shingles from text."""
        # Normalize text for shingling
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Create character shingles
        shingles = set()
        for i in range(len(normalized) - self.shingle_size + 1):
            shingle = normalized[i:i + self.shingle_size]
            shingles.add(shingle)
        
        return shingles
    
    def compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature for a set of shingles."""
        if not shingles:
            return [0] * self.num_hashes
        
        signature = []
        for hash_func in self.hash_functions:
            min_hash = min(hash_func(shingle) for shingle in shingles)
            signature.append(min_hash)
        
        return signature
    
    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def find_duplicates(self, passages: List[Dict[str, Any]]) -> List[Set[int]]:
        """Find groups of near-duplicate passages."""
        # Compute MinHash signatures
        signatures = []
        for passage in passages:
            shingles = self.create_shingles(passage['text'])
            signature = self.compute_minhash(shingles)
            signatures.append(signature)
        
        # Find similar passages
        duplicate_groups = []
        processed = set()
        
        for i in range(len(signatures)):
            if i in processed:
                continue
            
            current_group = {i}
            for j in range(i + 1, len(signatures)):
                if j in processed:
                    continue
                
                similarity = self.jaccard_similarity(signatures[i], signatures[j])
                if similarity >= self.threshold:
                    current_group.add(j)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed.update(current_group)
        
        return duplicate_groups
    
    def select_best_passage(self, passages: List[Dict[str, Any]], indices: Set[int]) -> int:
        """Select the best passage from a group of duplicates."""
        candidates = [(i, passages[i]) for i in indices]
        
        # Scoring criteria (higher is better)
        def score_passage(passage):
            score = 0
            
            # Prefer longer passages
            score += len(passage['text']) * 0.001
            
            # Prefer passages with more metadata
            if passage.get('title'):
                score += 10
            if passage.get('author'):
                score += 5
            if passage.get('date'):
                score += 5
            
            # Prefer passages from earlier in article (lower passage_index)
            passage_index = passage.get('passage_index', 0)
            score += max(0, 10 - passage_index)
            
            # Prefer passages with better text quality
            text = passage['text']
            if not text.endswith(('.', '!', '?')):
                score -= 5
            
            # Count sentences (more sentences = better structure)
            sentence_count = len([s for s in text.split('.') if s.strip()])
            score += min(sentence_count, 10)
            
            return score
        
        # Select passage with highest score
        best_idx = max(candidates, key=lambda x: score_passage(x[1]))[0]
        return best_idx
    
    def deduplicate(self, passages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Remove near-duplicate passages and return statistics."""
        if not passages:
            return passages, {'original_count': 0, 'deduplicated_count': 0, 'removed_count': 0}
        
        original_count = len(passages)
        
        # Find duplicate groups
        duplicate_groups = self.find_duplicates(passages)
        
        # Select best passage from each group
        indices_to_remove = set()
        for group in duplicate_groups:
            best_idx = self.select_best_passage(passages, group)
            # Remove all except the best one
            indices_to_remove.update(group - {best_idx})
        
        # Create deduplicated list
        deduplicated_passages = [
            passages[i] for i in range(len(passages)) 
            if i not in indices_to_remove
        ]
        
        # Statistics
        stats = {
            'original_count': original_count,
            'deduplicated_count': len(deduplicated_passages),
            'removed_count': len(indices_to_remove),
            'duplicate_groups': len(duplicate_groups),
            'reduction_percentage': (len(indices_to_remove) / original_count) * 100 if original_count > 0 else 0
        }
        
        return deduplicated_passages, stats

def deduplicate_passages(passages: List[Dict[str, Any]], 
                        threshold: float = 0.8) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience function for deduplicating passages."""
    deduplicator = MinHashDeduplicator(threshold=threshold)
    return deduplicator.deduplicate(passages)

if __name__ == "__main__":
    # Test deduplication
    sample_passages = [
        {
            'text': 'This is a sample passage about artificial intelligence and machine learning.',
            'passage_id': 'p1',
            'title': 'AI Article'
        },
        {
            'text': 'This is a sample passage about artificial intelligence and machine learning.',
            'passage_id': 'p2',
            'title': 'AI Article Copy'
        },
        {
            'text': 'This is a completely different passage about quantum computing.',
            'passage_id': 'p3',
            'title': 'Quantum Article'
        },
        {
            'text': 'This is a sample text about artificial intelligence and ML.',  # Similar to p1
            'passage_id': 'p4',
            'title': 'AI Article Variant'
        }
    ]
    
    deduplicator = MinHashDeduplicator(threshold=0.7)
    deduplicated, stats = deduplicator.deduplicate(sample_passages)
    
    print(f"Original passages: {stats['original_count']}")
    print(f"After deduplication: {stats['deduplicated_count']}")
    print(f"Removed: {stats['removed_count']} ({stats['reduction_percentage']:.1f}%)")
    print(f"Duplicate groups found: {stats['duplicate_groups']}")
    
    print("\nRemaining passages:")
    for passage in deduplicated:
        print(f"- {passage['passage_id']}: {passage['text'][:50]}...")