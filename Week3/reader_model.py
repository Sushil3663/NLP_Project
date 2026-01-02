"""
LSTM + Attention Reader Model for Extractive Question Answering.
Implements BiLSTM with bidirectional attention for answer span prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

class BiAttention(nn.Module):
    """Bidirectional attention mechanism between question and passage."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention weights (simplified)
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                question_hidden: torch.Tensor,
                passage_hidden: torch.Tensor,
                question_mask: torch.Tensor,
                passage_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional attention.
        
        Args:
            question_hidden: (batch_size, q_len, hidden_dim)
            passage_hidden: (batch_size, p_len, hidden_dim)
            question_mask: (batch_size, q_len)
            passage_mask: (batch_size, p_len)
        
        Returns:
            q2p_attended: (batch_size, p_len, hidden_dim) - question-to-passage attention
            p2q_attended: (batch_size, q_len, hidden_dim) - passage-to-question attention
        """
        batch_size, q_len, hidden_dim = question_hidden.shape
        p_len = passage_hidden.shape[1]
        
        # Compute attention scores
        # S[i,j] = alpha(q_i, p_j) where alpha is the attention function
        q_proj = self.w_q(question_hidden)  # (batch_size, q_len, hidden_dim)
        p_proj = self.w_p(passage_hidden)   # (batch_size, p_len, hidden_dim)
        
        # Compute pairwise attention scores using simpler approach
        # Compute similarity matrix: S[i,j] = q_i^T * p_j
        attention_scores = torch.bmm(q_proj, p_proj.transpose(1, 2))  # (batch_size, q_len, p_len)
        
        # Apply masks
        if question_mask is not None:
            question_mask_expanded = question_mask.unsqueeze(2)  # (batch_size, q_len, 1)
            attention_scores = attention_scores.masked_fill(question_mask_expanded == 0, -1e9)
        
        if passage_mask is not None:
            passage_mask_expanded = passage_mask.unsqueeze(1)  # (batch_size, 1, p_len)
            attention_scores = attention_scores.masked_fill(passage_mask_expanded == 0, -1e9)
        
        # Question-to-passage attention (for each passage position, attend to question)
        q2p_attention = F.softmax(attention_scores, dim=1)  # (batch_size, q_len, p_len)
        q2p_attention = self.dropout(q2p_attention)
        
        # Weighted sum of question representations for each passage position
        q2p_attended = torch.bmm(q2p_attention.transpose(1, 2), question_hidden)  # (batch_size, p_len, hidden_dim)
        
        # Passage-to-question attention (for each question position, attend to passage)
        p2q_attention = F.softmax(attention_scores, dim=2)  # (batch_size, q_len, p_len)
        p2q_attention = self.dropout(p2q_attention)
        
        # Weighted sum of passage representations for each question position
        p2q_attended = torch.bmm(p2q_attention, passage_hidden)  # (batch_size, q_len, hidden_dim)
        
        return q2p_attended, p2q_attended

class LSTMReader(nn.Module):
    """LSTM + Attention Reader for extractive question answering."""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Question encoder
        self.question_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Passage encoder
        self.passage_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Bidirectional attention
        self.bi_attention = BiAttention(lstm_output_dim, dropout)
        
        # Modeling layer (additional LSTM after attention)
        self.modeling_lstm = nn.LSTM(
            input_size=lstm_output_dim * 3,  # passage + q2p_attended + passage*q2p_attended
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Output layer for start/end predictions
        self.start_predictor = nn.Linear(lstm_output_dim, 1)
        self.end_predictor = nn.Linear(lstm_output_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padding tokens."""
        return (input_ids != 0).long()
    
    def forward(self, 
                question_ids: torch.Tensor,
                passage_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LSTM reader.
        
        Args:
            question_ids: (batch_size, q_len)
            passage_ids: (batch_size, p_len)
        
        Returns:
            start_logits: (batch_size, p_len)
            end_logits: (batch_size, p_len)
        """
        # Create masks
        question_mask = self.create_mask(question_ids)
        passage_mask = self.create_mask(passage_ids)
        
        # Embeddings
        question_emb = self.embedding(question_ids)  # (batch_size, q_len, embedding_dim)
        passage_emb = self.embedding(passage_ids)    # (batch_size, p_len, embedding_dim)
        
        question_emb = self.dropout(question_emb)
        passage_emb = self.dropout(passage_emb)
        
        # Encode question and passage
        question_hidden, _ = self.question_lstm(question_emb)  # (batch_size, q_len, hidden_dim*2)
        passage_hidden, _ = self.passage_lstm(passage_emb)     # (batch_size, p_len, hidden_dim*2)
        
        # Bidirectional attention
        q2p_attended, p2q_attended = self.bi_attention(
            question_hidden, passage_hidden, question_mask, passage_mask
        )
        
        # Combine passage representation with attention
        # passage_combined: passage + q2p_attended + passage * q2p_attended
        passage_combined = torch.cat([
            passage_hidden,
            q2p_attended,
            passage_hidden * q2p_attended
        ], dim=-1)  # (batch_size, p_len, hidden_dim*6)
        
        # Modeling layer
        modeled_passage, _ = self.modeling_lstm(passage_combined)  # (batch_size, p_len, hidden_dim*2)
        
        # Predict start and end positions
        start_logits = self.start_predictor(modeled_passage).squeeze(-1)  # (batch_size, p_len)
        end_logits = self.end_predictor(modeled_passage).squeeze(-1)      # (batch_size, p_len)
        
        # Apply passage mask to logits
        if passage_mask is not None:
            start_logits = start_logits.masked_fill(passage_mask == 0, -1e9)
            end_logits = end_logits.masked_fill(passage_mask == 0, -1e9)
        
        return start_logits, end_logits
    
    def predict_span(self, 
                    question_ids: torch.Tensor,
                    passage_ids: torch.Tensor,
                    max_answer_length: int = 30) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict answer spans with confidence scores.
        
        Args:
            question_ids: (batch_size, q_len)
            passage_ids: (batch_size, p_len)
            max_answer_length: Maximum allowed answer length
        
        Returns:
            start_positions: (batch_size,)
            end_positions: (batch_size,)
            confidence_scores: (batch_size,)
        """
        self.eval()
        
        with torch.no_grad():
            start_logits, end_logits = self.forward(question_ids, passage_ids)
            
            batch_size, seq_len = start_logits.shape
            
            # Convert logits to probabilities
            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)
            
            start_positions = []
            end_positions = []
            confidence_scores = []
            
            for i in range(batch_size):
                best_start = 0
                best_end = 0
                best_score = 0
                
                # Find best valid span
                for start_idx in range(seq_len):
                    for end_idx in range(start_idx, min(start_idx + max_answer_length, seq_len)):
                        score = start_probs[i, start_idx] * end_probs[i, end_idx]
                        if score > best_score:
                            best_score = score
                            best_start = start_idx
                            best_end = end_idx
                
                start_positions.append(best_start)
                end_positions.append(best_end)
                confidence_scores.append(best_score.item())
            
            return (torch.tensor(start_positions),
                   torch.tensor(end_positions),
                   torch.tensor(confidence_scores))

class SpanLoss(nn.Module):
    """Loss function for span prediction."""
    
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self,
                start_logits: torch.Tensor,
                end_logits: torch.Tensor,
                start_positions: torch.Tensor,
                end_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute span prediction loss.
        
        Args:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
            start_positions: (batch_size,)
            end_positions: (batch_size,)
        
        Returns:
            loss: scalar tensor
        """
        start_loss = self.cross_entropy(start_logits, start_positions)
        end_loss = self.cross_entropy(end_logits, end_positions)
        
        return (start_loss + end_loss) / 2

def extract_answer_text(passage_text: str, 
                       start_pos: int, 
                       end_pos: int,
                       tokenizer) -> str:
    """Extract answer text from passage using predicted positions."""
    tokens = tokenizer.tokenize(passage_text)
    
    if start_pos >= len(tokens) or end_pos >= len(tokens) or start_pos > end_pos:
        return ""
    
    answer_tokens = tokens[start_pos:end_pos + 1]
    return tokenizer.detokenize(answer_tokens)

if __name__ == "__main__":
    # Test the reader model
    vocab_size = 10000
    batch_size = 4
    q_len = 20
    p_len = 100
    
    # Create model
    model = LSTMReader(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2
    )
    
    # Test input
    question_ids = torch.randint(1, vocab_size, (batch_size, q_len))
    passage_ids = torch.randint(1, vocab_size, (batch_size, p_len))
    
    # Forward pass
    start_logits, end_logits = model(question_ids, passage_ids)
    
    print(f"Start logits shape: {start_logits.shape}")
    print(f"End logits shape: {end_logits.shape}")
    
    # Test span prediction
    start_pos, end_pos, confidence = model.predict_span(question_ids, passage_ids)
    
    print(f"Predicted spans:")
    for i in range(batch_size):
        print(f"  Sample {i}: start={start_pos[i]}, end={end_pos[i]}, confidence={confidence[i]:.3f}")
    
    # Test loss
    start_targets = torch.randint(0, p_len, (batch_size,))
    end_targets = torch.randint(0, p_len, (batch_size,))
    
    loss_fn = SpanLoss()
    loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
    
    print(f"Span loss: {loss.item():.4f}")