# Week 4: Inference Orchestration

## Purpose
Build the complete end-to-end inference pipeline that combines retrieval and reading for question answering with confidence scoring and fallback handling.

## Pipeline Architecture
```
Question Input → Preprocessing → Retrieval (top-20) → Re-ranking (top-5) → Reading → Answer + Citation
                     ↓              ↓                    ↓                ↓           ↓
              Entity/Date      BiLSTM Retriever    Cross-encoder    LSTM Reader   Confidence
              Extraction       + FAISS Index       Re-ranker        + Attention   Thresholding
```

## Components
- `inference_pipeline.py`: Main end-to-end QA pipeline
- `question_processor.py`: Question preprocessing and analysis
- `reranker.py`: Cross-encoder for passage re-ranking
- `answer_aggregator.py`: Answer span aggregation and confidence scoring
- `citation_formatter.py`: Citation and output formatting

## Pipeline Steps
1. **Question Preprocessing**: Extract entities, dates, question type
2. **Retrieval**: Get top-20 passages using BiLSTM retriever
3. **Re-ranking**: Re-rank to top-5 using cross-encoder
4. **Reading**: Extract answer spans from top passages
5. **Aggregation**: Combine and score candidate answers
6. **Output**: Format answer with citation and confidence

## Execution
```bash
cd Week4
python inference_pipeline.py "What is quantum computing?"
```

## Output
- Answer text with confidence score
- Citation: URL, headline, date, publisher
- Fallback: "not found" with top sources if confidence < threshold

## Success Criteria
- ✅ End-to-end pipeline functional
- ✅ Confidence thresholding implemented
- ✅ Citation formatting complete
- ✅ Fallback handling for low confidence