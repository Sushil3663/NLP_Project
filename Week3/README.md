# Week 3: LSTM + Attention Reader

## Purpose
Build and train an extractive reading comprehension model that finds answer spans in retrieved passages using BiLSTM with bidirectional attention.

## Architecture
```
Question → BiLSTM → Question Representation
    ↓
Passage → BiLSTM → Passage Representation
    ↓
Bidirectional Attention → Context-aware Representations
    ↓
Start/End Span Prediction → Answer Extraction
```

## Components
- `reader_model.py`: LSTM+attention reader architecture
- `train_reader.py`: Training pipeline with span prediction
- `squad_data.py`: SQuAD dataset preprocessing utilities
- `distant_supervision.py`: Create training data from news passages
- `evaluation.py`: EM and F1 score calculation

## Training Process
1. **Pretrain on SQuAD**: Train on SQuAD 2.0 for reading comprehension
2. **Distant Supervision**: Create training data from news passages
3. **Fine-tuning**: Adapt model to news domain
4. **Evaluation**: Compute EM and F1 scores

## Execution
```bash
cd Week3
python train_reader.py
```

## Output
- `../models/reader_model.pth`: Trained reader weights
- `../models/reader_tokenizer.pkl`: Text tokenizer
- `../outputs/week3_metrics.json`: Evaluation results
- `../outputs/week3_training_log.txt`: Training progress

## Success Criteria
- ✅ EM score > 65% on evaluation questions
- ✅ F1 score > 75% on evaluation questions
- ✅ Model handles "not found" cases appropriately