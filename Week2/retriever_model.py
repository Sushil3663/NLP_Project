"""
BiLSTM Retriever Model Architecture.
Encodes queries and passages into dense vectors for semantic similarity matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding to embeddings."""
    
    def __init__(self, embedding_dim: int, max_length: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           -(math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence representations."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention pooling to hidden states.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
        
        Returns:
            pooled: (batch_size, hidden_dim)
        """
        # Compute attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to hidden states
        pooled = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        
        return pooled

class BiLSTMEncoder(nn.Module):
    """BiLSTM encoder with attention pooling."""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_positional: bool = True,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_positional = use_positional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Positional encoding
        if use_positional:
            self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention pooling
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        self.attention_pooling = AttentionPooling(lstm_output_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padding tokens."""
        return (input_ids != 0).long()  # 1 for non-padding, 0 for padding
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BiLSTM encoder.
        
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:
            encoded: (batch_size, output_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask
        mask = self.create_mask(input_ids)
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Add positional encoding
        if self.use_positional:
            embedded = self.positional_encoding(embedded)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention pooling
        pooled = self.attention_pooling(lstm_out, mask)  # (batch_size, hidden_dim * 2)
        
        # Output projection
        output = self.output_projection(pooled)  # (batch_size, output_dim)
        output = self.layer_norm(output)
        
        return output

class DualEncoder(nn.Module):
    """Dual encoder for query and passage encoding."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 128,
                 output_dim: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 shared_encoder: bool = True,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.shared_encoder = shared_encoder
        self.output_dim = output_dim
        
        # Query encoder
        self.query_encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        # Passage encoder (shared or separate)
        if shared_encoder:
            self.passage_encoder = self.query_encoder
        else:
            self.passage_encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                pretrained_embeddings=pretrained_embeddings
            )
    
    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        """Encode query into dense vector."""
        return self.query_encoder(query_ids)
    
    def encode_passage(self, passage_ids: torch.Tensor) -> torch.Tensor:
        """Encode passage into dense vector."""
        return self.passage_encoder(passage_ids)
    
    def forward(self, query_ids: torch.Tensor, passage_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both query and passage.
        
        Args:
            query_ids: (batch_size, query_len)
            passage_ids: (batch_size, passage_len)
        
        Returns:
            query_vectors: (batch_size, output_dim)
            passage_vectors: (batch_size, output_dim)
        """
        query_vectors = self.encode_query(query_ids)
        passage_vectors = self.encode_passage(passage_ids)
        
        return query_vectors, passage_vectors

class TripletLoss(nn.Module):
    """Triplet loss for training dual encoder."""
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, 
                query_vectors: torch.Tensor,
                positive_vectors: torch.Tensor,
                negative_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            query_vectors: (batch_size, dim)
            positive_vectors: (batch_size, dim)
            negative_vectors: (batch_size, dim)
        
        Returns:
            loss: scalar tensor
        """
        # Compute similarities
        pos_sim = F.cosine_similarity(query_vectors, positive_vectors, dim=1)
        neg_sim = F.cosine_similarity(query_vectors, negative_vectors, dim=1)
        
        # Triplet loss
        loss = F.relu(self.margin - pos_sim + neg_sim)
        
        return loss.mean()

class ContrastiveLoss(nn.Module):
    """Contrastive loss for training dual encoder."""
    
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self,
                query_vectors: torch.Tensor,
                passage_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE).
        
        Args:
            query_vectors: (batch_size, dim)
            passage_vectors: (batch_size, dim)
        
        Returns:
            loss: scalar tensor
        """
        batch_size = query_vectors.size(0)
        
        # Normalize vectors
        query_vectors = F.normalize(query_vectors, p=2, dim=1)
        passage_vectors = F.normalize(passage_vectors, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_vectors, passage_vectors.T) / self.temperature
        
        # Labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=query_vectors.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

if __name__ == "__main__":
    # Test the model
    vocab_size = 10000
    batch_size = 4
    seq_len = 50
    
    # Create model
    model = DualEncoder(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        output_dim=512
    )
    
    # Test input
    query_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    passage_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    query_vectors, passage_vectors = model(query_ids, passage_ids)
    
    print(f"Query vectors shape: {query_vectors.shape}")
    print(f"Passage vectors shape: {passage_vectors.shape}")
    
    # Test loss
    triplet_loss = TripletLoss()
    negative_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    negative_vectors = model.encode_passage(negative_ids)
    
    loss = triplet_loss(query_vectors, passage_vectors, negative_vectors)
    print(f"Triplet loss: {loss.item()}")
    
    # Test contrastive loss
    contrastive_loss = ContrastiveLoss()
    loss = contrastive_loss(query_vectors, passage_vectors)
    print(f"Contrastive loss: {loss.item()}")