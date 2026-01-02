"""
Embedding utilities for loading and managing word embeddings.
Supports GloVe and FastText embeddings with vocabulary management.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import pickle
from tqdm import tqdm
import requests
import zipfile

class EmbeddingManager:
    """Manages word embeddings and vocabulary."""
    
    def __init__(self, embedding_dim: int = 300, max_vocab_size: int = 50000):
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
    
    def download_glove(self, glove_dir: str = "embeddings") -> str:
        """Download GloVe embeddings if not present."""
        os.makedirs(glove_dir, exist_ok=True)
        glove_file = os.path.join(glove_dir, "glove.6B.300d.txt")
        
        if os.path.exists(glove_file):
            print(f"GloVe embeddings found at {glove_file}")
            return glove_file
        
        print("Downloading GloVe embeddings...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_file = os.path.join(glove_dir, "glove.6B.zip")
        
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract the specific file we need
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extract("glove.6B.300d.txt", glove_dir)
            
            # Clean up zip file
            os.remove(zip_file)
            print(f"GloVe embeddings downloaded to {glove_file}")
            return glove_file
            
        except Exception as e:
            print(f"Error downloading GloVe: {e}")
            print("Please download GloVe manually from http://nlp.stanford.edu/data/glove.6B.zip")
            return None
    
    def load_glove_embeddings(self, glove_file: str, vocab: Optional[set] = None) -> bool:
        """Load GloVe embeddings from file."""
        if not os.path.exists(glove_file):
            print(f"GloVe file not found: {glove_file}")
            return False
        
        print(f"Loading GloVe embeddings from {glove_file}...")
        
        # First pass: count vocabulary and determine embedding dimension
        word_count = 0
        with open(glove_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().split()
            self.embedding_dim = len(first_line) - 1
            word_count = 1
            
            for line in f:
                word_count += 1
                if vocab and word_count > len(vocab) * 2:  # Early stopping if vocab provided
                    break
        
        print(f"Found {word_count} words, embedding dimension: {self.embedding_dim}")
        
        # Initialize vocabulary with special tokens
        self.word_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.idx_to_word = {i: token for i, token in enumerate(self.special_tokens)}
        current_idx = len(self.special_tokens)
        
        # Initialize embeddings matrix
        vocab_limit = min(self.max_vocab_size - len(self.special_tokens), word_count)
        total_vocab_size = vocab_limit + len(self.special_tokens)
        self.embeddings = np.random.normal(0, 0.1, (total_vocab_size, self.embedding_dim))
        
        # Set special token embeddings
        self.embeddings[self.word_to_idx[self.PAD_TOKEN]] = np.zeros(self.embedding_dim)
        
        # Second pass: load embeddings
        words_loaded = 0
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading embeddings", total=word_count):
                if words_loaded >= vocab_limit:
                    break
                
                parts = line.strip().split()
                if len(parts) != self.embedding_dim + 1:
                    continue
                
                word = parts[0]
                
                # Skip if vocab is provided and word not in vocab
                if vocab and word not in vocab:
                    continue
                
                try:
                    vector = np.array([float(x) for x in parts[1:]])
                    
                    if current_idx < total_vocab_size:
                        self.word_to_idx[word] = current_idx
                        self.idx_to_word[current_idx] = word
                        self.embeddings[current_idx] = vector
                        current_idx += 1
                        words_loaded += 1
                
                except ValueError:
                    continue
        
        self.vocab_size = current_idx
        self.embeddings = self.embeddings[:self.vocab_size]
        
        print(f"Loaded {words_loaded} word embeddings")
        print(f"Final vocabulary size: {self.vocab_size}")
        return True
    
    def create_simple_embeddings(self, vocab: set) -> bool:
        """Create simple random embeddings if GloVe is not available."""
        print("Creating simple random embeddings...")
        
        # Build vocabulary
        self.word_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        self.idx_to_word = {i: token for i, token in enumerate(self.special_tokens)}
        current_idx = len(self.special_tokens)
        
        # Add vocabulary words
        for word in sorted(vocab):
            if current_idx >= self.max_vocab_size:
                break
            self.word_to_idx[word] = current_idx
            self.idx_to_word[current_idx] = word
            current_idx += 1
        
        self.vocab_size = current_idx
        
        # Create random embeddings
        self.embeddings = np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
        self.embeddings[self.word_to_idx[self.PAD_TOKEN]] = np.zeros(self.embedding_dim)
        
        print(f"Created {self.vocab_size} random embeddings")
        return True
    
    def text_to_indices(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to list of word indices."""
        words = text.lower().split()
        indices = []
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx[self.UNK_TOKEN])
        
        # Truncate or pad if max_length specified
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.word_to_idx[self.PAD_TOKEN]] * (max_length - len(indices)))
        
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """Convert list of indices back to text."""
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                    words.append(word)
        return ' '.join(words)
    
    def get_embedding_layer(self, trainable: bool = False) -> nn.Embedding:
        """Create PyTorch embedding layer."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded!")
        
        embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, 
                                     padding_idx=self.word_to_idx[self.PAD_TOKEN])
        
        # Initialize with pre-trained embeddings
        embedding_layer.weight.data.copy_(torch.from_numpy(self.embeddings))
        
        # Set trainable
        embedding_layer.weight.requires_grad = trainable
        
        return embedding_layer
    
    def save(self, filepath: str):
        """Save embedding manager to file."""
        data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'embeddings': self.embeddings,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved embedding manager to {filepath}")
    
    def load(self, filepath: str) -> bool:
        """Load embedding manager from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.embeddings = data['embeddings']
            self.vocab_size = data['vocab_size']
            self.embedding_dim = data['embedding_dim']
            self.special_tokens = data['special_tokens']
            
            print(f"Loaded embedding manager from {filepath}")
            return True
        
        except Exception as e:
            print(f"Error loading embedding manager: {e}")
            return False

def build_vocabulary_from_passages(passages: List[Dict]) -> set:
    """Build vocabulary from passage texts."""
    vocab = set()
    
    for passage in tqdm(passages, desc="Building vocabulary"):
        # Add words from passage text
        text = passage.get('text', '')
        words = text.lower().split()
        vocab.update(words)
        
        # Add words from title
        title = passage.get('title', '')
        if title:
            title_words = title.lower().split()
            vocab.update(title_words)
    
    print(f"Built vocabulary with {len(vocab)} unique words")
    return vocab

if __name__ == "__main__":
    # Test embedding manager
    manager = EmbeddingManager(embedding_dim=50)  # Small for testing
    
    # Create simple vocabulary
    test_vocab = {'hello', 'world', 'test', 'embedding', 'neural', 'network'}
    
    # Create embeddings
    manager.create_simple_embeddings(test_vocab)
    
    # Test text conversion
    text = "hello world test"
    indices = manager.text_to_indices(text)
    recovered_text = manager.indices_to_text(indices)
    
    print(f"Original: {text}")
    print(f"Indices: {indices}")
    print(f"Recovered: {recovered_text}")
    
    # Test embedding layer
    embedding_layer = manager.get_embedding_layer()
    print(f"Embedding layer: {embedding_layer}")
    print(f"Embedding shape: {embedding_layer.weight.shape}")