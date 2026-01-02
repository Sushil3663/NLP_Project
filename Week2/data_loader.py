"""
Data loader for BiLSTM retriever training.
Handles triplet sampling and batch preparation for training.
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import os

class RetrieverDataset(Dataset):
    """Dataset for training BiLSTM retriever with triplet loss."""
    
    def __init__(self,
                 passages: List[Dict[str, Any]],
                 queries: List[Dict[str, Any]],
                 embedding_manager,
                 max_query_length: int = 64,
                 max_passage_length: int = 256,
                 hard_negatives_ratio: float = 0.5):
        
        self.passages = passages
        self.queries = queries
        self.embedding_manager = embedding_manager
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.hard_negatives_ratio = hard_negatives_ratio
        
        # Build passage lookup
        self.passage_id_to_idx = {p['passage_id']: i for i, p in enumerate(passages)}
        self.article_to_passages = defaultdict(list)
        
        for i, passage in enumerate(passages):
            article_id = passage.get('article_id', '')
            self.article_to_passages[article_id].append(i)
        
        # Prepare training triplets
        self.triplets = self._create_triplets()
        
        print(f"Created {len(self.triplets)} training triplets")
    
    def _create_triplets(self) -> List[Dict[str, Any]]:
        """Create (query, positive, negative) triplets for training."""
        triplets = []
        
        for query in self.queries:
            positive_passage_id = query.get('positive_passage_id')
            if positive_passage_id not in self.passage_id_to_idx:
                continue
            
            positive_idx = self.passage_id_to_idx[positive_passage_id]
            article_id = query.get('article_id', '')
            
            # Get negative candidates (from different articles)
            negative_candidates = []
            for i, passage in enumerate(self.passages):
                if passage.get('article_id', '') != article_id:
                    negative_candidates.append(i)
            
            if not negative_candidates:
                continue
            
            # Create multiple triplets per query for data augmentation
            num_negatives = min(3, len(negative_candidates))
            selected_negatives = random.sample(negative_candidates, num_negatives)
            
            for negative_idx in selected_negatives:
                triplet = {
                    'query': query['query'],
                    'positive_idx': positive_idx,
                    'negative_idx': negative_idx,
                    'query_type': query.get('type', 'unknown')
                }
                triplets.append(triplet)
        
        return triplets
    
    def _get_hard_negatives(self, query_vector: torch.Tensor, 
                           exclude_indices: set, k: int = 10) -> List[int]:
        """Get hard negative passages using approximate nearest neighbor search."""
        # This is a simplified version - in practice, you'd use the current model
        # to find hard negatives during training
        
        candidates = [i for i in range(len(self.passages)) if i not in exclude_indices]
        return random.sample(candidates, min(k, len(candidates)))
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        triplet = self.triplets[idx]
        
        # Get query
        query_text = triplet['query']
        query_ids = self.embedding_manager.text_to_indices(query_text, self.max_query_length)
        
        # Get positive passage
        positive_passage = self.passages[triplet['positive_idx']]
        positive_text = positive_passage['text']
        positive_ids = self.embedding_manager.text_to_indices(positive_text, self.max_passage_length)
        
        # Get negative passage
        negative_passage = self.passages[triplet['negative_idx']]
        negative_text = negative_passage['text']
        negative_ids = self.embedding_manager.text_to_indices(negative_text, self.max_passage_length)
        
        return {
            'query_ids': torch.tensor(query_ids, dtype=torch.long),
            'positive_ids': torch.tensor(positive_ids, dtype=torch.long),
            'negative_ids': torch.tensor(negative_ids, dtype=torch.long),
            'query_text': query_text,
            'positive_text': positive_text,
            'negative_text': negative_text
        }

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning (InfoNCE loss)."""
    
    def __init__(self,
                 passages: List[Dict[str, Any]],
                 queries: List[Dict[str, Any]],
                 embedding_manager,
                 max_query_length: int = 64,
                 max_passage_length: int = 256):
        
        self.passages = passages
        self.queries = queries
        self.embedding_manager = embedding_manager
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        
        # Build passage lookup
        self.passage_id_to_idx = {p['passage_id']: i for i, p in enumerate(passages)}
        
        # Filter queries with valid positive passages
        self.valid_queries = []
        for query in queries:
            positive_passage_id = query.get('positive_passage_id')
            if positive_passage_id in self.passage_id_to_idx:
                self.valid_queries.append(query)
        
        print(f"Created contrastive dataset with {len(self.valid_queries)} queries")
    
    def __len__(self) -> int:
        return len(self.valid_queries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.valid_queries[idx]
        
        # Get query
        query_text = query['query']
        query_ids = self.embedding_manager.text_to_indices(query_text, self.max_query_length)
        
        # Get positive passage
        positive_passage_id = query['positive_passage_id']
        positive_idx = self.passage_id_to_idx[positive_passage_id]
        positive_passage = self.passages[positive_idx]
        positive_text = positive_passage['text']
        positive_ids = self.embedding_manager.text_to_indices(positive_text, self.max_passage_length)
        
        return {
            'query_ids': torch.tensor(query_ids, dtype=torch.long),
            'passage_ids': torch.tensor(positive_ids, dtype=torch.long),
            'query_text': query_text,
            'passage_text': positive_text
        }

def collate_triplets(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for triplet batch."""
    query_ids = torch.stack([item['query_ids'] for item in batch])
    positive_ids = torch.stack([item['positive_ids'] for item in batch])
    negative_ids = torch.stack([item['negative_ids'] for item in batch])
    
    return {
        'query_ids': query_ids,
        'positive_ids': positive_ids,
        'negative_ids': negative_ids,
        'query_texts': [item['query_text'] for item in batch],
        'positive_texts': [item['positive_text'] for item in batch],
        'negative_texts': [item['negative_text'] for item in batch]
    }

def collate_contrastive(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for contrastive batch."""
    query_ids = torch.stack([item['query_ids'] for item in batch])
    passage_ids = torch.stack([item['passage_ids'] for item in batch])
    
    return {
        'query_ids': query_ids,
        'passage_ids': passage_ids,
        'query_texts': [item['query_text'] for item in batch],
        'passage_texts': [item['passage_text'] for item in batch]
    }

def create_data_loaders(passages: List[Dict[str, Any]],
                       queries: List[Dict[str, Any]],
                       embedding_manager,
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       loss_type: str = 'triplet') -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    # Split queries into train/val
    random.shuffle(queries)
    split_idx = int(len(queries) * train_split)
    train_queries = queries[:split_idx]
    val_queries = queries[split_idx:]
    
    print(f"Train queries: {len(train_queries)}")
    print(f"Validation queries: {len(val_queries)}")
    
    if loss_type == 'triplet':
        # Create triplet datasets
        train_dataset = RetrieverDataset(passages, train_queries, embedding_manager)
        val_dataset = RetrieverDataset(passages, val_queries, embedding_manager)
        collate_fn = collate_triplets
    
    elif loss_type == 'contrastive':
        # Create contrastive datasets
        train_dataset = ContrastiveDataset(passages, train_queries, embedding_manager)
        val_dataset = ContrastiveDataset(passages, val_queries, embedding_manager)
        collate_fn = collate_contrastive
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def load_data(data_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load passages and queries from data directory."""
    
    # Load passages
    passages_file = os.path.join(data_dir, 'processed_passages.json')
    with open(passages_file, 'r', encoding='utf-8') as f:
        passages = json.load(f)
    
    # Load queries
    queries_file = os.path.join(data_dir, 'training_queries.json')
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"Loaded {len(passages)} passages and {len(queries)} queries")
    return passages, queries

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from embeddings import EmbeddingManager
    
    # Test data loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    if os.path.exists(os.path.join(data_dir, 'processed_passages.json')):
        passages, queries = load_data(data_dir)
        
        # Create simple embedding manager for testing
        embedding_manager = EmbeddingManager(embedding_dim=100)
        vocab = set()
        for passage in passages[:100]:  # Test with subset
            vocab.update(passage['text'].lower().split())
        for query in queries[:100]:
            vocab.update(query['query'].lower().split())
        
        embedding_manager.create_simple_embeddings(vocab)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            passages[:100], queries[:100], embedding_manager, 
            batch_size=4, loss_type='triplet'
        )
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Query IDs shape: {batch['query_ids'].shape}")
        print(f"Positive IDs shape: {batch['positive_ids'].shape}")
        print(f"Negative IDs shape: {batch['negative_ids'].shape}")
    
    else:
        print("No processed data found. Run Week1 data processing first.")