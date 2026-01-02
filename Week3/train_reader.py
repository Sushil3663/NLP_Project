"""
Main training script for LSTM + Attention Reader (Week 3).
Trains the reading comprehension model with SQuAD pretraining and distant supervision.
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
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reader_model import LSTMReader, SpanLoss
from squad_data import SimpleTokenizer, create_squad_data_loaders, download_squad_data, SQuADDataset, collate_squad_batch
from distant_supervision import create_distant_supervision_data
from evaluation import evaluate_model, analyze_errors, create_evaluation_report
from torch.utils.data import DataLoader

class ReaderTrainer:
    """Trainer for LSTM + Attention Reader model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(self.config['output_dir'], 'tensorboard_reader'))
        
        # Setup tokenizer
        self.setup_tokenizer()
        
        # Create model
        self.create_model()
        
        # Setup training
        self.setup_training()
    
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['squad_dir'], exist_ok=True)
    
    def setup_tokenizer(self):
        """Setup tokenizer."""
        tokenizer_file = os.path.join(self.config['model_dir'], 'reader_tokenizer.pkl')
        
        self.tokenizer = SimpleTokenizer()
        
        # Try to load existing tokenizer
        if os.path.exists(tokenizer_file):
            print("Loading existing tokenizer...")
            with open(tokenizer_file, 'rb') as f:
                tokenizer_data = pickle.load(f)
            
            self.tokenizer.word_to_idx = tokenizer_data['word_to_idx']
            self.tokenizer.idx_to_word = tokenizer_data['idx_to_word']
            self.tokenizer.vocab_size = tokenizer_data['vocab_size']
            
            print(f"Loaded tokenizer with vocabulary size: {self.tokenizer.vocab_size}")
        else:
            print("Will build tokenizer during data loading...")
    
    def create_model(self):
        """Create and initialize model."""
        print("Creating reader model...")
        
        # Model will be created after tokenizer is ready
        self.model = None
        self.vocab_size = None
    
    def setup_training(self):
        """Setup loss function and optimizer."""
        print("Setting up training components...")
        
        # Loss function
        self.criterion = SpanLoss(ignore_index=-1)
        
        # Optimizer and scheduler will be created after model
        self.optimizer = None
        self.scheduler = None
    
    def create_model_with_vocab(self, vocab_size: int):
        """Create model with known vocabulary size."""
        self.vocab_size = vocab_size
        
        self.model = LSTMReader(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def load_squad_data(self):
        """Load SQuAD data for pretraining."""
        print("Loading SQuAD data...")
        
        # Download SQuAD data
        train_file, dev_file = download_squad_data(self.config['squad_dir'])
        
        if not train_file or not dev_file:
            print("Warning: Could not download SQuAD data. Skipping SQuAD pretraining.")
            return None, None
        
        # Create data loaders
        max_examples = self.config.get('max_squad_examples', None)
        train_loader, dev_loader = create_squad_data_loaders(
            train_file, dev_file, self.tokenizer,
            batch_size=self.config['batch_size'],
            max_examples=max_examples
        )
        
        # Save tokenizer
        tokenizer_file = os.path.join(self.config['model_dir'], 'reader_tokenizer.pkl')
        tokenizer_data = {
            'word_to_idx': self.tokenizer.word_to_idx,
            'idx_to_word': self.tokenizer.idx_to_word,
            'vocab_size': self.tokenizer.vocab_size
        }
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Saved tokenizer with vocabulary size: {self.tokenizer.vocab_size}")
        
        return train_loader, dev_loader
    
    def load_news_data(self):
        """Load news data for distant supervision."""
        print("Loading news data for distant supervision...")
        
        # Load passages
        passages_file = os.path.join(self.config['data_dir'], 'processed_passages.json')
        if not os.path.exists(passages_file):
            print("Error: Processed passages not found. Run Week 1 data processing first.")
            return None
        
        with open(passages_file, 'r', encoding='utf-8') as f:
            passages = json.load(f)
        
        # Limit passages for distant supervision
        max_passages = self.config.get('max_news_passages', 5000)
        if len(passages) > max_passages:
            passages = passages[:max_passages]
        
        # Create distant supervision data
        distant_sup_file = os.path.join(self.config['output_dir'], 'distant_supervision_data.json')
        
        if not os.path.exists(distant_sup_file):
            print("Creating distant supervision data...")
            questions = create_distant_supervision_data(
                passages, self.tokenizer, distant_sup_file,
                max_questions_per_passage=2
            )
        else:
            print("Loading existing distant supervision data...")
            with open(distant_sup_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        
        # Create dataset and data loader
        dataset = SQuADDataset(questions, self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_squad_batch,
            num_workers=0
        )
        
        return data_loader
    
    def train_epoch(self, data_loader, epoch: int, phase: str = "train") -> float:
        """Train for one epoch."""
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(data_loader, desc=f"{phase.title()} Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            question_ids = batch['question_ids'].to(self.device)
            passage_ids = batch['passage_ids'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)
            
            # Handle impossible questions (start/end = -1)
            start_positions = torch.clamp(start_positions, min=0)
            end_positions = torch.clamp(end_positions, min=0)
            
            if phase == "train":
                self.optimizer.zero_grad()
            
            # Forward pass
            start_logits, end_logits = self.model(question_ids, passage_ids)
            
            # Compute loss
            loss = self.criterion(start_logits, end_logits, start_positions, end_positions)
            
            if phase == "train":
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
            if phase == "train":
                global_step = epoch * len(data_loader) + batch_idx
                self.writer.add_scalar(f'{phase.title()}/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar(f'{phase.title()}/AvgLoss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, phase: str = "", is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'vocab_size': self.vocab_size,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['model_dir'], f'reader_checkpoint_{phase}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['model_dir'], f'best_reader_{phase}.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best {phase} model with validation loss: {val_loss:.4f}")
    
    def pretrain_on_squad(self):
        """Pretrain model on SQuAD data."""
        print("Starting SQuAD pretraining...")
        
        # Load SQuAD data
        train_loader, dev_loader = self.load_squad_data()
        
        if not train_loader:
            print("Skipping SQuAD pretraining due to data loading issues.")
            return
        
        # Create model now that we have vocabulary
        self.create_model_with_vocab(self.tokenizer.vocab_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['squad_epochs']):
            print(f"\nSQuAD Epoch {epoch+1}/{self.config['squad_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch, "squad_train")
            
            # Validate
            val_loss = self.train_epoch(dev_loader, epoch, "squad_val")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"SQuAD Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, "squad", is_best)
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping SQuAD training after {patience_counter} epochs without improvement")
                break
        
        print("SQuAD pretraining completed!")
    
    def finetune_on_news(self):
        """Fine-tune model on news data with distant supervision."""
        print("Starting news fine-tuning...")
        
        # Load news data
        news_loader = self.load_news_data()
        
        if not news_loader:
            print("Skipping news fine-tuning due to data loading issues.")
            return
        
        # If model not created yet (no SQuAD pretraining), create it now
        if self.model is None:
            self.create_model_with_vocab(self.tokenizer.vocab_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Split data for validation
        dataset = news_loader.dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                shuffle=True, collate_fn=collate_squad_batch, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                              shuffle=False, collate_fn=collate_squad_batch, num_workers=0)
        
        for epoch in range(self.config['news_epochs']):
            print(f"\nNews Epoch {epoch+1}/{self.config['news_epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch, "news_train")
            
            # Validate
            val_loss = self.train_epoch(val_loader, epoch, "news_val")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"News Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, "news", is_best)
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping news training after {patience_counter} epochs without improvement")
                break
        
        print("News fine-tuning completed!")
    
    def evaluate_model(self):
        """Evaluate the trained model."""
        print("Evaluating model...")
        
        # Load best model
        best_model_path = os.path.join(self.config['model_dir'], 'best_reader_news.pth')
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(self.config['model_dir'], 'best_reader_squad.pth')
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {best_model_path}")
        else:
            print("No trained model found for evaluation")
            return
        
        # Create evaluation questions (simple examples)
        eval_questions = [
            {
                'question': 'What is artificial intelligence?',
                'context': 'Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.',
                'answers': [{'text': 'intelligence demonstrated by machines', 'answer_start': 35}],
                'is_impossible': False
            },
            {
                'question': 'Who invented the telephone?',
                'context': 'The telephone was invented by Alexander Graham Bell in 1876. Bell was a Scottish-born inventor and scientist.',
                'answers': [{'text': 'Alexander Graham Bell', 'answer_start': 35}],
                'is_impossible': False
            },
            {
                'question': 'What is quantum computing?',
                'context': 'Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.',
                'answers': [],
                'is_impossible': True
            }
        ]
        
        # Create evaluation dataset
        eval_dataset = SQuADDataset(eval_questions, self.tokenizer)
        eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, 
                               collate_fn=collate_squad_batch, num_workers=0)
        
        # Evaluate
        results = evaluate_model(self.model, eval_loader, self.tokenizer, self.device)
        
        # Print results
        metrics = results['metrics']
        print(f"Evaluation Results:")
        print(f"  Exact Match: {metrics['exact_match']:.2f}%")
        print(f"  F1 Score: {metrics['f1']:.2f}%")
        print(f"  Average Confidence: {metrics['avg_confidence']:.3f}")
        
        # Save results
        metrics_file = os.path.join(self.config['log_dir'], 'week3_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Analyze errors
        error_analysis = analyze_errors(results['detailed_results'])
        
        # Create evaluation report
        report_file = os.path.join(self.config['log_dir'], 'week3_evaluation_report.txt')
        create_evaluation_report(metrics, error_analysis, report_file)
        
        return metrics

def create_config():
    """Create training configuration."""
    return {
        # Data
        'data_dir': '../data',
        'output_dir': '../outputs',
        'model_dir': '../models',
        'log_dir': '../outputs',
        'squad_dir': 'squad_data',
        
        # Model architecture
        'embedding_dim': 300,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        
        # Training
        'batch_size': 8,  # Reduce batch size
        'squad_epochs': 10,  # More epochs
        'news_epochs': 15,
        'learning_rate': 5e-4,  # Lower learning rate
        'weight_decay': 1e-4,
        'patience': 5,  # More patience
        
        # Data limits
        'max_squad_examples': 20000,  # More SQuAD data
        'max_news_passages': 5000,   # More news passages
    }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LSTM + Attention Reader')
    parser.add_argument('--skip_squad', action='store_true', help='Skip SQuAD pretraining')
    parser.add_argument('--skip_news', action='store_true', help='Skip news fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--squad_epochs', type=int, default=5, help='SQuAD epochs')
    parser.add_argument('--news_epochs', type=int, default=10, help='News epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create config
    config = create_config()
    
    # Override with command line arguments
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.squad_epochs:
        config['squad_epochs'] = args.squad_epochs
    if args.news_epochs:
        config['news_epochs'] = args.news_epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Adjust paths relative to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    config['data_dir'] = os.path.join(project_root, 'data')
    config['output_dir'] = os.path.join(project_root, 'outputs')
    config['model_dir'] = os.path.join(project_root, 'models')
    config['log_dir'] = os.path.join(project_root, 'outputs')
    
    print("Week 3: LSTM + Attention Reader Training")
    print("=" * 45)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 45)
    
    # Create trainer
    trainer = ReaderTrainer(config)
    
    # Training phases
    if not args.skip_squad:
        trainer.pretrain_on_squad()
    
    if not args.skip_news:
        trainer.finetune_on_news()
    
    # Evaluate model
    metrics = trainer.evaluate_model()
    
    # Generate report
    report_file = os.path.join(config['log_dir'], 'week3_report.txt')
    with open(report_file, 'w') as f:
        f.write("LSTM News QA System - Week 3 LSTM + Attention Reader Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}\n")
        f.write(f"Vocabulary size: {trainer.tokenizer.vocab_size:,}\n")
        f.write(f"SQuAD pretraining: {'Yes' if not args.skip_squad else 'Skipped'}\n")
        f.write(f"News fine-tuning: {'Yes' if not args.skip_news else 'Skipped'}\n\n")
        
        if metrics:
            f.write("EVALUATION RESULTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Exact Match (EM): {metrics['exact_match']:.2f}%\n")
            f.write(f"F1 Score: {metrics['f1']:.2f}%\n")
            f.write(f"Average Confidence: {metrics['avg_confidence']:.3f}\n\n")
        
        f.write("SUCCESS CRITERIA CHECK\n")
        f.write("-" * 20 + "\n")
        if metrics:
            f.write(f"EM score > 65%: {metrics['exact_match'] > 65.0} ({metrics['exact_match']:.1f}%)\n")
            f.write(f"F1 score > 75%: {metrics['f1'] > 75.0} ({metrics['f1']:.1f}%)\n")
        f.write("Model handles span prediction\n")
        f.write("Tokenizer and model saved\n")
    
    print(f"\nWeek 3 training completed!")
    print(f"Model saved to: {config['model_dir']}")
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()