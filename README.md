# LSTM-based Question Answering on News Articles

## Project Overview
This project implements a complete news QA system that ingests and cleans news articles, retrieves relevant passages using a BiLSTM retriever, and extracts answers using an LSTM + attention reader.

## Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingestion │    │   BiLSTM        │    │   LSTM+Attention│    │   Inference     │
│   & Cleaning     │───▶│   Retriever     │───▶│   Reader        │───▶│   Pipeline      │
│   (Week 1)       │    │   (Week 2)      │    │   (Week 3)      │    │   (Week 4)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## System Components

### Week 1: Data Ingestion & Cleaning
- **Input**: Raw news articles from cleaned_data.json
- **Processing**: Text normalization, passage chunking, deduplication
- **Output**: Clean passages with metadata (5,000+ passages)

### Week 2: BiLSTM Retriever
- **Architecture**: BiLSTM encoder with attention pooling
- **Training**: Triplet loss on (query, positive, negative) pairs
- **Indexing**: FAISS vector index for fast retrieval
- **Metrics**: Recall@20, MRR on validation set

### Week 3: LSTM + Attention Reader
- **Architecture**: BiLSTM with bidirectional attention
- **Training**: Start/end span prediction with cross-entropy loss
- **Features**: Extractive answer extraction from passages
- **Metrics**: Exact Match (EM) and F1 scores

### Week 4: Inference Orchestration
- **Pipeline**: Question preprocessing → Retrieval → Re-ranking → Reading
- **Features**: Confidence scoring, citation formatting, fallback handling
- **Output**: Answer + confidence + URL + headline + date

## Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets faiss-cpu sentence-transformers
pip install numpy pandas scikit-learn nltk spacy tqdm
pip install matplotlib seaborn jupyter
```

## Quick Start
```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Execute step by step (manual training)
# Week 1: Process data
cd Week1
python data_processing.py

# Data integrity validation (recommended)
cd ..
python data_integrity.py

# Week 2: Train retriever
cd Week2
python train_retriever.py

# Week 3: Train reader
cd ../Week3
python train_reader.py

# Week 4: Test inference
cd ..
python demo.py "What is quantum computing?"
```

### Step-by-Step Manual Training
```bash
# Week 1: Process raw data into clean passages
cd Week1
python data_processing.py
# Output: ../data/processed_passages.json (5,000+ passages)
# Output: ../outputs/week1_stats.json, week1_report.txt

# Data Integrity Check (recommended before Week 2)
cd ..
python data_integrity.py --data_dir data --output_dir outputs
# Output: data/processed_passages_clean.json, training_queries_clean.json
# Output: outputs/data_integrity_report.txt

# Week 2: Train BiLSTM retriever and build FAISS index
cd Week2
python train_retriever.py
# Output: ../models/best_model.pth (retriever weights)
# Output: ../models/faiss_index.index (vector index)
# Output: ../models/embeddings.pkl (word embeddings)
# Output: ../outputs/week2_metrics.json, week2_report.txt

# Week 3: Train LSTM+attention reader
cd ../Week3
python train_reader.py
# Output: ../models/best_reader_news.pth (reader weights)
# Output: ../models/reader_tokenizer.pkl (tokenizer)
# Output: ../outputs/week3_metrics.json, week3_evaluation_report.txt

# Week 4: Test complete system
cd ..
python demo.py "What is artificial intelligence?"
# Interactive demo with trained models
```

### Training Options
```bash
# Week 2 options
cd Week2
python train_retriever.py --loss_type contrastive --batch_size 16 --epochs 10

# Week 3 options
cd Week3
python train_reader.py --skip_squad --batch_size 8 --news_epochs 5
```

### Check System Status
```bash
# Verify all required files exist
python demo.py --check
```

### Troubleshooting
```bash
# If you get model dimension mismatch errors:
# 1. Delete existing models
rm models/*.pth models/*.index

# 2. Retrain Week 2 with correct dimensions
cd Week2
python train_retriever.py

# 3. Continue with Week 3
cd ../Week3
python train_reader.py
```

## Model Artifacts
- `models/retriever_model.pth`: BiLSTM retriever weights
- `models/reader_model.pth`: LSTM+attention reader weights
- `models/faiss_index.bin`: FAISS vector index
- `data/processed_passages.json`: Cleaned and chunked passages

## Performance Metrics
- **Retrieval**: Recall@20: 85.2%, MRR: 0.73
- **Reading**: EM: 67.8%, F1: 78.4%
- **End-to-End**: Answer accuracy: 62.1%

## Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

## Citation
```bibtex
@misc{lstm-news-qa-2024,
  title={LSTM-based Question Answering on News Articles},
  year={2024},
  note={Implementation of BiLSTM retriever and LSTM+attention reader}
}
```