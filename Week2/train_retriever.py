"""
Main training script for BiLSTM retriever (Week 2).
Trains the dual encoder model and builds FAISS index for retrieval.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever_model import DualEncoder, TripletLoss, ContrastiveLoss
from data_loader import create_data_loaders, load_data
from embeddings import EmbeddingManager, build_vocabulary_from_passages
from indexing import build_retrieval_index, evaluate_retrieval, FAISSIndex

class RetrieverTrainer:
    """Trainer for BiLSTM retriever model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(self.config['output_dir'], 'tensorboard'))
        
        # Load data
        self.load_data()
        
        # Setup embeddings
        self.setup_embeddings()
        
        # Create model
        self.create_model()
        
        # Setup training
        self.setup_training()
    
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
    
    def load_data(self):
        """Load passages and queries with integrity validation."""
        print("Loading and validating data...")
        
        # Run data integrity validation
        sys.path.append(os.path.dirname(self.config['data_dir']))
        from data_integrity import DataIntegrityValidator
        
        validator = DataIntegrityValidator(self.config['data_dir'], self.config['output_dir'])
        validation_result = validator.validate_and_fix(auto_fix=True)
        
        # Use cleaned and aligned data
        aligned_splits = validation_result['aligned_splits']
        
        # Use ALL passages for index building (critical fix)
        self.passages = aligned_splits['all_passages']
        self.queries = aligned_splits['all_queries']
        
        # Use aligned splits for training/validation
        self.train_passages = aligned_splits['train_passages']
        self.val_passages = aligned_splits['val_passages']
        self.train_queries = aligned_splits['train_queries']
        self.val_queries = aligned_splits['val_queries']
        
        print(f"Data integrity validation completed:")
        print(f"  Total passages: {len(self.passages)}")
        print(f"  Total queries: {len(self.queries)}")
        print(f"  Training passages: {len(self.train_passages)}")
        print(f"  Validation passages: {len(self.val_passages)}")
        print(f"  Training queries: {len(self.train_queries)}")
        print(f"  Validation queries: {len(self.val_queries)}")
        
        # Verify alignment
        train_passage_ids = {p.get('passage_id') for p in self.train_passages}
        val_passage_ids = {p.get('passage_id') for p in self.val_passages}
        
        train_query_positives = {q.get('positive_passage_id') for q in self.train_queries}
        val_query_positives = {q.get('positive_passage_id') for q in self.val_queries}
        
        # Check alignment
        train_aligned = train_query_positives.issubset(train_passage_ids)
        val_aligned = val_query_positives.issubset(val_passage_ids)
        
        print(f"  Train alignment: {'✅' if train_aligned else '❌'}")
        print(f"  Val alignment: {'✅' if val_aligned else '❌'}")
        
        if not (train_aligned and val_aligned):
            raise ValueError("Data alignment failed. Check data integrity report.")
    
    def setup_embeddings(self):
        """Setup word embeddings."""
        print("Setting up embeddings...")
        
        embedding_file = os.path.join(self.config['model_dir'], 'embeddings.pkl')
        self.embedding_manager = EmbeddingManager(
            embedding_dim=self.config['embedding_dim'],
            max_vocab_size=self.config['max_vocab_size']
        )
        
        # Try to load existing embeddings
        if os.path.exists(embedding_file):
            print("Loading existing embeddings...")
            self.embedding_manager.load(embedding_file)
        else:
            print("Creating new embeddings...")
            
            # Build vocabulary
            vocab = build_vocabulary_from_passages(self.passages)
            
            # Try to load GloVe embeddings
            glove_file = self.embedding_manager.download_glove()
            if glove_file and os.path.exists(glove_file):
                success = self.embedding_manager.load_glove_embeddings(glove_file, vocab)
                if not success:
                    self.embedding_manager.create_simple_embeddings(vocab)
            else:
                self.embedding_manager.create_simple_embeddings(vocab)
            
            # Save embeddings
            self.embedding_manager.save(embedding_file)
        
        print(f"Vocabulary size: {self.embedding_manager.vocab_size}")
    
    def create_model(self):
        """Create and initialize model."""
        print("Creating model...")
        
        # Get pretrained embeddings
        pretrained_embeddings = torch.from_numpy(self.embedding_manager.embeddings).float()
        
        self.model = DualEncoder(
            vocab_size=self.embedding_manager.vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            shared_encoder=self.config['shared_encoder'],
            pretrained_embeddings=pretrained_embeddings
        )
        
        self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """Setup loss function, optimizer, and data loaders."""
        print("Setting up training...")
        
        # Loss function
        if self.config['loss_type'] == 'triplet':
            self.criterion = TripletLoss(margin=self.config['margin'])
        elif self.config['loss_type'] == 'contrastive':
            self.criterion = ContrastiveLoss(temperature=self.config['temperature'])
        else:
            raise ValueError(f"Unknown loss type: {self.config['loss_type']}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config['patience']
        )
        
        # Data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            self.train_passages,
            self.train_queries,  # Use aligned train queries
            self.embedding_manager,
            batch_size=self.config['batch_size'],
            train_split=0.8,
            loss_type=self.config['loss_type']
        )
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config['loss_type'] == 'triplet':
                query_vectors = self.model.encode_query(batch['query_ids'])
                positive_vectors = self.model.encode_passage(batch['positive_ids'])
                negative_vectors = self.model.encode_passage(batch['negative_ids'])
                
                loss = self.criterion(query_vectors, positive_vectors, negative_vectors)
            
            elif self.config['loss_type'] == 'contrastive':
                query_vectors, passage_vectors = self.model(batch['query_ids'], batch['passage_ids'])
                loss = self.criterion(query_vectors, passage_vectors)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        return total_loss / num_batches
    
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                if self.config['loss_type'] == 'triplet':
                    query_vectors = self.model.encode_query(batch['query_ids'])
                    positive_vectors = self.model.encode_passage(batch['positive_ids'])
                    negative_vectors = self.model.encode_passage(batch['negative_ids'])
                    
                    loss = self.criterion(query_vectors, positive_vectors, negative_vectors)
                
                elif self.config['loss_type'] == 'contrastive':
                    query_vectors, passage_vectors = self.model(batch['query_ids'], batch['passage_ids'])
                    loss = self.criterion(query_vectors, passage_vectors)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['model_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['model_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Training for {self.config['num_epochs']} epochs")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training log
        training_log = []
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log results
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            training_log.append(log_entry)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Save training log
        log_file = os.path.join(self.config['log_dir'], 'training_log.json')
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print("Training completed!")
        return best_val_loss
    
    def build_index_and_evaluate(self):
        """Build FAISS index and evaluate retrieval performance."""
        print("Building FAISS index and evaluating...")
        
        # Load best model
        best_model_path = os.path.join(self.config['model_dir'], 'best_model.pth')
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Build index using ALL passages (critical fix)
        # This ensures validation queries can find their positive passages
        index_path = os.path.join(self.config['model_dir'], 'faiss_index')
        faiss_index = build_retrieval_index(
            self.model,
            self.passages,  # Use ALL passages, not just train_passages
            self.embedding_manager,
            index_path,
            batch_size=self.config['batch_size'],
            index_type='flat',
            device=self.device
        )
        
        # Evaluate on validation queries (now properly aligned)
        if self.val_queries:
            print(f"Evaluating on {len(self.val_queries)} validation queries...")
            
            # Verify all positive passages exist in index
            all_passage_ids = {p.get('passage_id') for p in self.passages}
            val_positives = {q.get('positive_passage_id') for q in self.val_queries}
            missing_positives = val_positives - all_passage_ids
            
            if missing_positives:
                print(f"Warning: {len(missing_positives)} positive passages missing from index")
                print(f"Missing IDs: {list(missing_positives)[:5]}...")  # Show first 5
            else:
                print("✅ All validation query positives found in index")
            
            metrics = evaluate_retrieval(
                self.model,
                faiss_index,
                self.val_queries,
                self.embedding_manager,
                k=20,
                device=self.device
            )
            
            print(f"Retrieval Evaluation Results:")
            print(f"  Recall@20: {metrics['recall_at_k']:.3f}")
            print(f"  MRR: {metrics['mrr']:.3f}")
            print(f"  Valid queries: {metrics['valid_queries']}")
            print(f"  Total queries: {len(self.val_queries)}")
            
            # Save metrics
            metrics_file = os.path.join(self.config['log_dir'], 'week2_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            print("No validation queries available for evaluation")
        
        return faiss_index

def create_config():
    """Create training configuration."""
    return {
        # Data
        'data_dir': '../data',
        'output_dir': '../outputs',
        'model_dir': '../models',
        'log_dir': '../outputs',
        
        # Model architecture
        'embedding_dim': 300,
        'hidden_dim': 128,  # Keep original lower dimensions
        'output_dim': 128,  # Keep original lower dimensions
        'num_layers': 1,
        'dropout': 0.1,
        'shared_encoder': True,
        'max_vocab_size': 50000,
        
        # Training
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 5,
        
        # Loss function
        'loss_type': 'triplet',  # 'triplet' or 'contrastive'
        'margin': 0.2,  # for triplet loss
        'temperature': 0.05,  # for contrastive loss
    }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BiLSTM Retriever')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--loss_type', type=str, choices=['triplet', 'contrastive'], 
                       default='triplet', help='Loss function type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create config
    config = create_config()
    
    # Override with command line arguments
    if args.loss_type:
        config['loss_type'] = args.loss_type
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Adjust paths relative to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    config['data_dir'] = os.path.join(project_root, 'data')
    config['output_dir'] = os.path.join(project_root, 'outputs')
    config['model_dir'] = os.path.join(project_root, 'models')
    config['log_dir'] = os.path.join(project_root, 'outputs')
    
    # Check if data exists
    if not os.path.exists(os.path.join(config['data_dir'], 'processed_passages.json')):
        print("Error: Processed data not found!")
        print("Please run Week 1 data processing first:")
        print("cd Week1 && python data_processing.py")
        return
    
    print("Week 2: BiLSTM Retriever Training")
    print("=" * 40)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 40)
    
    # Create trainer and train
    trainer = RetrieverTrainer(config)
    best_val_loss = trainer.train()
    
    # Build index and evaluate
    faiss_index = trainer.build_index_and_evaluate()
    
    # Generate report
    report_file = os.path.join(config['log_dir'], 'week2_report.txt')
    with open(report_file, 'w') as f:
        f.write("LSTM News QA System - Week 2 BiLSTM Retriever Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}\n")
        f.write(f"Training passages: {len(trainer.train_passages):,}\n")
        f.write(f"Vocabulary size: {trainer.embedding_manager.vocab_size:,}\n\n")
        
        f.write("SUCCESS CRITERIA CHECK\n")
        f.write("-" * 20 + "\n")
        f.write("BiLSTM retriever trained successfully\n")
        f.write("FAISS index built and saved\n")
        f.write("Model checkpoints saved\n")
    
    print(f"\nWeek 2 training completed successfully!")
    print(f"Model saved to: {config['model_dir']}")
    print(f"Index saved to: {os.path.join(config['model_dir'], 'faiss_index')}")
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()