"""
SQuAD dataset preprocessing utilities for pretraining the reader model.
Downloads and processes SQuAD 2.0 data for reading comprehension training.
"""

import json
import os
import requests
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import re

class SimpleTokenizer:
    """Simple word-level tokenizer for text processing."""
    
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        text = text.lower().strip()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        return ' '.join(tokens)
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 50000):
        """Build vocabulary from texts."""
        word_counts = {}
        
        for text in tqdm(texts, desc="Building vocabulary"):
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:max_vocab_size - 2]:  # -2 for PAD and UNK
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        print(f"Built vocabulary with {self.vocab_size} words")
    
    def text_to_indices(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to list of indices."""
        tokens = self.tokenize(text)
        indices = []
        
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # Truncate or pad if max_length specified
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.word_to_idx['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def find_answer_span(self, passage_text: str, answer_text: str) -> Tuple[int, int]:
        """Find answer span in passage tokens."""
        passage_tokens = self.tokenize(passage_text)
        answer_tokens = self.tokenize(answer_text)
        
        if not answer_tokens:
            return -1, -1
        
        # Find the answer span in passage tokens
        for i in range(len(passage_tokens) - len(answer_tokens) + 1):
            if passage_tokens[i:i + len(answer_tokens)] == answer_tokens:
                return i, i + len(answer_tokens) - 1
        
        # If exact match not found, try fuzzy matching
        for i in range(len(passage_tokens)):
            for j in range(i, min(i + len(answer_tokens) + 5, len(passage_tokens))):
                candidate = ' '.join(passage_tokens[i:j + 1])
                if answer_text.lower() in candidate.lower() or candidate.lower() in answer_text.lower():
                    return i, j
        
        return -1, -1

class SQuADDataset(Dataset):
    """Dataset for SQuAD reading comprehension data."""
    
    def __init__(self,
                 data: List[Dict[str, Any]],
                 tokenizer: SimpleTokenizer,
                 max_question_length: int = 64,
                 max_passage_length: int = 512):
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length
        
        # Filter valid examples
        self.valid_examples = []
        for example in data:
            try:
                # Handle different field names for context/passage
                context = example.get('context') or example.get('passage') or example.get('text', '')
                question = example.get('question', '')
                
                if not context or not question:
                    continue
                
                if example.get('is_impossible', False):
                    # For impossible questions
                    self.valid_examples.append({
                        'question': question,
                        'passage': context,
                        'start_position': -1,
                        'end_position': -1,
                        'answer_text': '',
                        'is_impossible': True
                    })
                elif example.get('answers'):
                    # For answerable questions
                    answer = example['answers'][0]  # Take first answer
                    answer_text = answer.get('text', '')
                    
                    if not answer_text:
                        continue
                    
                    # Find answer span in passage
                    start_pos, end_pos = tokenizer.find_answer_span(context, answer_text)
                    
                    if start_pos != -1 and end_pos != -1:
                        self.valid_examples.append({
                            'question': question,
                            'passage': context,
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'answer_text': answer_text,
                            'is_impossible': False
                        })
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
        
        print(f"Created SQuAD dataset with {len(self.valid_examples)} valid examples")
    
    def __len__(self) -> int:
        return len(self.valid_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.valid_examples[idx]
        
        # Tokenize question and passage
        question_ids = self.tokenizer.text_to_indices(example['question'], self.max_question_length)
        passage_ids = self.tokenizer.text_to_indices(example['passage'], self.max_passage_length)
        
        # Adjust positions for truncation
        start_pos = example['start_position']
        end_pos = example['end_position']
        
        if start_pos >= self.max_passage_length:
            start_pos = -1
            end_pos = -1
        elif end_pos >= self.max_passage_length:
            end_pos = self.max_passage_length - 1
        
        return {
            'question_ids': torch.tensor(question_ids, dtype=torch.long),
            'passage_ids': torch.tensor(passage_ids, dtype=torch.long),
            'start_position': torch.tensor(start_pos, dtype=torch.long),
            'end_position': torch.tensor(end_pos, dtype=torch.long),
            'question_text': example['question'],
            'passage_text': example['passage'],
            'answer_text': example['answer_text'],
            'is_impossible': example['is_impossible']
        }

def download_squad_data(data_dir: str = "squad_data") -> Tuple[str, str]:
    """Download SQuAD 2.0 dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    urls = {
        'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
        'dev': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
    }
    
    filepaths = {}
    
    for split, url in urls.items():
        filepath = os.path.join(data_dir, f'{split}-v2.0.json')
        
        if os.path.exists(filepath):
            print(f"SQuAD {split} data already exists at {filepath}")
            filepaths[split] = filepath
            continue
        
        print(f"Downloading SQuAD {split} data...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {split}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            filepaths[split] = filepath
            print(f"Downloaded {split} data to {filepath}")
            
        except Exception as e:
            print(f"Error downloading {split} data: {e}")
            filepaths[split] = None
    
    return filepaths.get('train'), filepaths.get('dev')

def load_squad_data(filepath: str) -> List[Dict[str, Any]]:
    """Load and parse SQuAD data from JSON file."""
    if not filepath or not os.path.exists(filepath):
        print(f"SQuAD file not found: {filepath}")
        return []
    
    print(f"Loading SQuAD data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                question = qa['question']
                qas_id = qa['id']
                is_impossible = qa.get('is_impossible', False)
                
                example = {
                    'id': qas_id,
                    'question': question,
                    'context': context,
                    'is_impossible': is_impossible,
                    'answers': []
                }
                
                if not is_impossible and 'answers' in qa:
                    for answer in qa['answers']:
                        example['answers'].append({
                            'text': answer['text'],
                            'answer_start': answer['answer_start']
                        })
                
                examples.append(example)
    
    print(f"Loaded {len(examples)} examples from SQuAD")
    return examples

def create_squad_data_loaders(train_file: str,
                             dev_file: str,
                             tokenizer: SimpleTokenizer,
                             batch_size: int = 32,
                             max_examples: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """Create SQuAD data loaders for training and validation."""
    
    # Load data
    train_data = load_squad_data(train_file)
    dev_data = load_squad_data(dev_file)
    
    if max_examples:
        train_data = train_data[:max_examples]
        dev_data = dev_data[:max_examples // 10]
    
    # Build vocabulary from training data
    all_texts = []
    for example in train_data:
        all_texts.append(example['question'])
        all_texts.append(example['context'])
    
    tokenizer.build_vocab(all_texts)
    
    # Create datasets
    train_dataset = SQuADDataset(train_data, tokenizer)
    dev_dataset = SQuADDataset(dev_data, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_squad_batch,
        num_workers=0
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_squad_batch,
        num_workers=0
    )
    
    return train_loader, dev_loader

def collate_squad_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for SQuAD batch."""
    question_ids = torch.stack([item['question_ids'] for item in batch])
    passage_ids = torch.stack([item['passage_ids'] for item in batch])
    start_positions = torch.stack([item['start_position'] for item in batch])
    end_positions = torch.stack([item['end_position'] for item in batch])
    
    return {
        'question_ids': question_ids,
        'passage_ids': passage_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'question_texts': [item['question_text'] for item in batch],
        'passage_texts': [item['passage_text'] for item in batch],
        'answer_texts': [item['answer_text'] for item in batch],
        'is_impossible': [item['is_impossible'] for item in batch]
    }

if __name__ == "__main__":
    # Test SQuAD data loading
    data_dir = "squad_data"
    
    # Download data
    train_file, dev_file = download_squad_data(data_dir)
    
    if train_file and dev_file:
        # Create tokenizer
        tokenizer = SimpleTokenizer()
        
        # Create data loaders (with limited examples for testing)
        train_loader, dev_loader = create_squad_data_loaders(
            train_file, dev_file, tokenizer, batch_size=4, max_examples=1000
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Dev batches: {len(dev_loader)}")
        
        # Test batch
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Question IDs shape: {batch['question_ids'].shape}")
        print(f"Passage IDs shape: {batch['passage_ids'].shape}")
        print(f"Start positions: {batch['start_positions']}")
        print(f"End positions: {batch['end_positions']}")
        
        # Test tokenizer
        sample_text = "What is the capital of France?"
        tokens = tokenizer.tokenize(sample_text)
        indices = tokenizer.text_to_indices(sample_text)
        print(f"Sample text: {sample_text}")
        print(f"Tokens: {tokens}")
        print(f"Indices: {indices}")
    
    else:
        print("Failed to download SQuAD data")