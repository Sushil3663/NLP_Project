"""
FAISS indexing utilities for fast passage retrieval.
Builds and manages vector indices for semantic search.
"""

import os
import numpy as np
import torch
import faiss
import pickle
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import json

class FAISSIndex:
    """FAISS-based vector index for passage retrieval."""
    
    def __init__(self, 
                 embedding_dim: int,
                 index_type: str = 'flat',
                 nlist: int = 100,
                 nprobe: int = 10):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('flat', 'ivf', 'hnsw')
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to search in IVF index
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.passage_ids = []
        self.passage_metadata = []
        self.is_trained = False
    
    def _create_index(self, num_vectors: int) -> faiss.Index:
        """Create appropriate FAISS index based on type and size."""
        
        if self.index_type == 'flat':
            # Exact search using L2 distance
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        elif self.index_type == 'ivf':
            # Inverted file index for faster approximate search
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            index.nprobe = self.nprobe
        
        elif self.index_type == 'hnsw':
            # Hierarchical Navigable Small World for very fast search
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def build_index(self, 
                   embeddings: np.ndarray,
                   passage_ids: List[str],
                   passage_metadata: List[Dict[str, Any]]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: (num_passages, embedding_dim) numpy array
            passage_ids: List of passage IDs
            passage_metadata: List of passage metadata dicts
        """
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = self._create_index(len(embeddings))
        
        # Train index if needed (for IVF)
        if self.index_type == 'ivf':
            print("Training IVF index...")
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add vectors to index
        print("Adding vectors to index...")
        self.index.add(embeddings)
        
        # Store metadata
        self.passage_ids = passage_ids
        self.passage_metadata = passage_metadata
        
        print(f"Index built successfully. Total vectors: {self.index.ntotal}")
    
    def search(self, 
               query_embeddings: np.ndarray,
               k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar passages.
        
        Args:
            query_embeddings: (num_queries, embedding_dim) numpy array
            k: Number of results to return per query
        
        Returns:
            scores: (num_queries, k) similarity scores
            indices: (num_queries, k) passage indices
        """
        if self.index is None:
            raise ValueError("Index not built yet!")
        
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        
        return scores, indices
    
    def get_passages(self, indices: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        Get passage metadata for search results.
        
        Args:
            indices: (num_queries, k) array of passage indices
        
        Returns:
            List of lists of passage metadata
        """
        results = []
        
        for query_indices in indices:
            query_results = []
            for idx in query_indices:
                if 0 <= idx < len(self.passage_metadata):
                    passage = self.passage_metadata[idx].copy()
                    passage['passage_id'] = self.passage_ids[idx]
                    query_results.append(passage)
                else:
                    # Handle invalid indices
                    query_results.append({
                        'passage_id': 'invalid',
                        'text': '',
                        'title': '',
                        'url': '',
                        'score': 0.0
                    })
            results.append(query_results)
        
        return results
    
    def save(self, filepath: str):
        """Save FAISS index and metadata to disk."""
        # Save FAISS index
        index_file = filepath + '.index'
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'passage_ids': self.passage_ids,
            'passage_metadata': self.passage_metadata,
            'is_trained': self.is_trained
        }
        
        metadata_file = filepath + '.metadata'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved FAISS index to {index_file}")
        print(f"Saved metadata to {metadata_file}")
    
    def load(self, filepath: str) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            index_file = filepath + '.index'
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            metadata_file = filepath + '.metadata'
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            self.nlist = metadata['nlist']
            self.nprobe = metadata['nprobe']
            self.passage_ids = metadata['passage_ids']
            self.passage_metadata = metadata['passage_metadata']
            self.is_trained = metadata['is_trained']
            
            print(f"Loaded FAISS index from {index_file}")
            print(f"Index contains {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False

def encode_passages_batch(model, 
                         passages: List[Dict[str, Any]], 
                         embedding_manager,
                         batch_size: int = 32,
                         max_length: int = 256,
                         device: str = 'cuda') -> np.ndarray:
    """
    Encode passages in batches using the trained model.
    
    Args:
        model: Trained BiLSTM encoder
        passages: List of passage dictionaries
        embedding_manager: Embedding manager for text preprocessing
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        device: Device to run model on
    
    Returns:
        embeddings: (num_passages, embedding_dim) numpy array
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
            batch_passages = passages[i:i + batch_size]
            
            # Prepare batch
            batch_texts = [p['text'] for p in batch_passages]
            batch_ids = []
            
            for text in batch_texts:
                ids = embedding_manager.text_to_indices(text, max_length)
                batch_ids.append(ids)
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch_ids, dtype=torch.long).to(device)
            
            # Encode
            embeddings = model.encode_passage(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

def build_retrieval_index(model,
                         passages: List[Dict[str, Any]],
                         embedding_manager,
                         output_path: str,
                         batch_size: int = 32,
                         index_type: str = 'flat',
                         device: str = 'cuda') -> FAISSIndex:
    """
    Build complete retrieval index from trained model and passages.
    
    Args:
        model: Trained BiLSTM encoder
        passages: List of passage dictionaries
        embedding_manager: Embedding manager
        output_path: Path to save index
        batch_size: Batch size for encoding
        index_type: Type of FAISS index
        device: Device to run model on
    
    Returns:
        Built FAISS index
    """
    print("Building retrieval index...")
    
    # Encode all passages
    embeddings = encode_passages_batch(
        model, passages, embedding_manager, 
        batch_size=batch_size, device=device
    )
    
    # Prepare metadata
    passage_ids = [p['passage_id'] for p in passages]
    passage_metadata = passages
    
    # Create and build index
    embedding_dim = embeddings.shape[1]
    faiss_index = FAISSIndex(embedding_dim, index_type=index_type)
    faiss_index.build_index(embeddings, passage_ids, passage_metadata)
    
    # Save index
    faiss_index.save(output_path)
    
    return faiss_index

def evaluate_retrieval(model,
                      faiss_index: FAISSIndex,
                      queries: List[Dict[str, Any]],
                      embedding_manager,
                      k: int = 20,
                      device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate retrieval performance using Recall@k and MRR.
    
    Args:
        model: Trained BiLSTM encoder
        faiss_index: Built FAISS index
        queries: List of query dictionaries
        embedding_manager: Embedding manager
        k: Number of results to retrieve
        device: Device to run model on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    recall_at_k = 0
    mrr_sum = 0
    valid_queries = 0
    
    with torch.no_grad():
        for query in tqdm(queries, desc="Evaluating retrieval"):
            # Encode query
            query_text = query['query']
            query_ids = embedding_manager.text_to_indices(query_text, 64)
            query_tensor = torch.tensor([query_ids], dtype=torch.long).to(device)
            query_embedding = model.encode_query(query_tensor).cpu().numpy()
            
            # Search
            scores, indices = faiss_index.search(query_embedding, k)
            retrieved_passage_ids = [faiss_index.passage_ids[idx] for idx in indices[0]]
            
            # Check if positive passage is retrieved
            positive_passage_id = query.get('positive_passage_id')
            if positive_passage_id:
                valid_queries += 1
                
                if positive_passage_id in retrieved_passage_ids:
                    recall_at_k += 1
                    
                    # Calculate reciprocal rank
                    rank = retrieved_passage_ids.index(positive_passage_id) + 1
                    mrr_sum += 1.0 / rank
    
    if valid_queries == 0:
        return {'recall_at_k': 0.0, 'mrr': 0.0, 'valid_queries': 0}
    
    metrics = {
        'recall_at_k': recall_at_k / valid_queries,
        'mrr': mrr_sum / valid_queries,
        'valid_queries': valid_queries,
        'k': k
    }
    
    return metrics

if __name__ == "__main__":
    # Test FAISS index
    embedding_dim = 128
    num_passages = 1000
    
    # Create dummy embeddings
    embeddings = np.random.randn(num_passages, embedding_dim).astype(np.float32)
    passage_ids = [f"passage_{i}" for i in range(num_passages)]
    passage_metadata = [{'text': f'This is passage {i}', 'title': f'Title {i}'} 
                       for i in range(num_passages)]
    
    # Build index
    faiss_index = FAISSIndex(embedding_dim, index_type='flat')
    faiss_index.build_index(embeddings, passage_ids, passage_metadata)
    
    # Test search
    query_embeddings = np.random.randn(5, embedding_dim).astype(np.float32)
    scores, indices = faiss_index.search(query_embeddings, k=10)
    
    print(f"Search results shape: {scores.shape}, {indices.shape}")
    print(f"Top scores for first query: {scores[0][:5]}")
    
    # Test save/load
    test_path = "test_index"
    faiss_index.save(test_path)
    
    new_index = FAISSIndex(embedding_dim)
    success = new_index.load(test_path)
    print(f"Load successful: {success}")
    
    # Clean up test files
    import os
    for ext in ['.index', '.metadata']:
        if os.path.exists(test_path + ext):
            os.remove(test_path + ext)