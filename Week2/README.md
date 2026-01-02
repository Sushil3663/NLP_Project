# Week 2: BiLSTM Retriever

## Purpose
Build and train a BiLSTM-based passage retriever that encodes queries and passages into dense vectors for semantic similarity matching.

## Architecture
```
Input Text → Word Embeddings → BiLSTM → Attention Pooling → Dense Vector
                ↓
         [GloVe/FastText + Positional + Date Features]
```

## Components
- `retriever_model.py`: BiLSTM encoder architecture
- `train_retriever.py`: Training pipeline with triplet loss
- `data_loader.py`: Data loading and batch preparation
- `embeddings.py`: Word embedding utilities (GloVe/FastText)
- `indexing.py`: FAISS index creation and management

## Training Process
1. **Data Preparation**: Create (query, positive, negative) triplets
2. **Embedding**: Load pre-trained word embeddings
3. **Model Training**: BiLSTM with triplet/contrastive loss
4. **Indexing**: Build FAISS index for fast retrieval
5. **Evaluation**: Recall@20 and MRR metrics

## Execution
```bash
cd Week2
python train_retriever.py
```

## Output
- `../models/retriever_model.pth`: Trained BiLSTM weights
- `../models/faiss_index.bin`: FAISS vector index
- `../outputs/week2_metrics.json`: Evaluation results
- `../outputs/week2_training_log.txt`: Training progress

## Success Criteria
- ✅ Recall@20 > 80% on validation set
- ✅ MRR > 0.70 on validation queries
- ✅ FAISS index with all passages indexed