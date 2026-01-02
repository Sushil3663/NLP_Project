"""
Cross-encoder re-ranker for improving passage ranking.
Uses a smaller BiLSTM model to re-rank retrieved passages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np

class CrossEncoder(nn.Module):
    """Cross-encoder for question-passage relevance scoring."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 pretrained_embeddings: torch.Tensor = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Attention pooling
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Cross-attention between question and passage
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final scoring layers
        self.scorer = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, hidden_dim),  # *2 for question + passage
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def create_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padding tokens."""
        return (input_ids != 0).long()
    
    def encode_sequence(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence with BiLSTM."""
        mask = self.create_mask(input_ids)
        
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention pooling
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        pooled = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        return lstm_out, pooled
    
    def forward(self, 
                question_ids: torch.Tensor,
                passage_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for relevance scoring.
        
        Args:
            question_ids: (batch_size, q_len)
            passage_ids: (batch_size, p_len)
        
        Returns:
            scores: (batch_size,) relevance scores between 0 and 1
        """
        # Encode question and passage
        question_hidden, question_pooled = self.encode_sequence(question_ids)
        passage_hidden, passage_pooled = self.encode_sequence(passage_ids)
        
        # Cross-attention between question and passage
        # Use question as query, passage as key/value
        attended_passage, _ = self.cross_attention(
            question_hidden, passage_hidden, passage_hidden
        )
        
        # Pool attended passage
        attended_passage_pooled = torch.mean(attended_passage, dim=1)
        
        # Combine representations
        combined = torch.cat([question_pooled, attended_passage_pooled], dim=-1)
        
        # Score relevance
        scores = self.scorer(combined).squeeze(-1)
        
        return scores

class PassageReranker:
    """Re-ranks retrieved passages using cross-encoder."""
    
    def __init__(self, 
                 model: CrossEncoder,
                 embedding_manager,
                 device: str = 'cuda',
                 max_question_length: int = 64,
                 max_passage_length: int = 256):
        
        self.model = model
        self.embedding_manager = embedding_manager
        self.device = device
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length
        
        self.model.to(device)
        self.model.eval()
    
    def rerank_passages(self, 
                       question: str,
                       passages: List[Dict[str, Any]],
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank passages based on relevance to question.
        
        Args:
            question: Input question
            passages: List of passage dictionaries
            top_k: Number of top passages to return
        
        Returns:
            Re-ranked list of passages with scores
        """
        if not passages:
            return []
        
        # Prepare batch data
        batch_size = len(passages)
        question_ids = self.embedding_manager.text_to_indices(question, self.max_question_length)
        
        question_batch = []
        passage_batch = []
        
        for passage in passages:
            passage_text = passage.get('text', '')
            passage_ids = self.embedding_manager.text_to_indices(passage_text, self.max_passage_length)
            
            question_batch.append(question_ids)
            passage_batch.append(passage_ids)
        
        # Convert to tensors
        question_tensor = torch.tensor(question_batch, dtype=torch.long).to(self.device)
        passage_tensor = torch.tensor(passage_batch, dtype=torch.long).to(self.device)
        
        # Get relevance scores
        with torch.no_grad():
            scores = self.model(question_tensor, passage_tensor)
            scores = scores.cpu().numpy()
        
        # Add scores to passages and sort
        scored_passages = []
        for i, passage in enumerate(passages):
            passage_copy = passage.copy()
            passage_copy['rerank_score'] = float(scores[i])
            scored_passages.append(passage_copy)
        
        # Sort by score (descending)
        scored_passages.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top-k
        return scored_passages[:top_k]

def create_simple_reranker(embedding_manager, device: str = 'cuda') -> PassageReranker:
    """Create a simple re-ranker using the embedding manager's vocabulary."""
    
    # Create a simple cross-encoder model
    model = CrossEncoder(
        vocab_size=embedding_manager.vocab_size,
        embedding_dim=embedding_manager.embedding_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        pretrained_embeddings=torch.from_numpy(embedding_manager.embeddings).float()
    )
    
    # Initialize with reasonable weights (since we don't have trained weights)
    # In practice, this would be trained on relevance data
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    return PassageReranker(model, embedding_manager, device)

def train_reranker_simple(model: CrossEncoder,
                         training_data: List[Dict[str, Any]],
                         embedding_manager,
                         num_epochs: int = 5,
                         batch_size: int = 16,
                         learning_rate: float = 1e-3,
                         device: str = 'cuda') -> CrossEncoder:
    """
    Simple training procedure for the re-ranker.
    Training data should contain: question, positive_passage, negative_passage
    """
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    print(f"Training re-ranker for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Simple batch processing
        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i + batch_size]
            
            if len(batch_data) < 2:  # Need at least 2 examples
                continue
            
            # Prepare positive and negative examples
            questions = []
            passages = []
            labels = []
            
            for example in batch_data:
                question = example['question']
                pos_passage = example['positive_passage']
                neg_passage = example.get('negative_passage', '')
                
                # Add positive example
                questions.append(embedding_manager.text_to_indices(question, 64))
                passages.append(embedding_manager.text_to_indices(pos_passage, 256))
                labels.append(1.0)
                
                # Add negative example if available
                if neg_passage:
                    questions.append(embedding_manager.text_to_indices(question, 64))
                    passages.append(embedding_manager.text_to_indices(neg_passage, 256))
                    labels.append(0.0)
            
            if len(questions) < 2:
                continue
            
            # Convert to tensors
            question_tensor = torch.tensor(questions, dtype=torch.long).to(device)
            passage_tensor = torch.tensor(passages, dtype=torch.long).to(device)
            label_tensor = torch.tensor(labels, dtype=torch.float).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            scores = model(question_tensor, passage_tensor)
            loss = criterion(scores, label_tensor)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    model.eval()
    return model

if __name__ == "__main__":
    # Test the cross-encoder
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from Week2.embeddings import EmbeddingManager
    
    # Create simple embedding manager for testing
    embedding_manager = EmbeddingManager(embedding_dim=100)
    test_vocab = {'hello', 'world', 'test', 'question', 'passage', 'artificial', 'intelligence', 'computer', 'science'}
    embedding_manager.create_simple_embeddings(test_vocab)
    
    # Create re-ranker
    reranker = create_simple_reranker(embedding_manager, device='cpu')
    
    # Test data
    question = "What is artificial intelligence?"
    passages = [
        {
            'text': 'Artificial intelligence is intelligence demonstrated by machines.',
            'title': 'AI Definition',
            'url': 'http://example.com/ai',
            'score': 0.8
        },
        {
            'text': 'Computer science is the study of computational systems.',
            'title': 'Computer Science',
            'url': 'http://example.com/cs',
            'score': 0.6
        },
        {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'title': 'Machine Learning',
            'url': 'http://example.com/ml',
            'score': 0.7
        }
    ]
    
    # Re-rank passages
    reranked = reranker.rerank_passages(question, passages, top_k=3)
    
    print("Re-ranking Test Results:")
    print(f"Question: {question}")
    print("\nRe-ranked passages:")
    
    for i, passage in enumerate(reranked, 1):
        print(f"{i}. {passage['title']} (score: {passage['rerank_score']:.3f})")
        print(f"   Original score: {passage['score']}")
        print(f"   Text: {passage['text'][:60]}...")
        print()