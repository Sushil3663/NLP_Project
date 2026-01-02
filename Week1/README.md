# Week 1: Data Ingestion & Cleaning

## Purpose
Process raw news articles into clean, chunked passages suitable for retrieval and reading tasks.

## Components
- `data_processing.py`: Main data processing pipeline
- `text_utils.py`: Text cleaning and normalization utilities
- `chunking.py`: Passage chunking with overlap
- `deduplication.py`: Near-duplicate passage removal

## Pipeline Steps
1. **Load Data**: Read articles from cleaned_data.json
2. **Text Cleaning**: Remove boilerplate, normalize Unicode, fix formatting
3. **Passage Chunking**: Split into 200-400 token passages with 50-token overlap
4. **Deduplication**: Remove near-identical passages using MinHash
5. **Metadata Extraction**: Extract publisher, date, entities

## Execution
```bash
cd Week1
python data_processing.py
```

## Output
- `../data/processed_passages.json`: Clean passages with metadata
- `../outputs/week1_stats.json`: Processing statistics
- `../outputs/week1_report.txt`: Detailed processing report

## Success Criteria
- ✅ At least 5,000 passages generated
- ✅ Duplicates reduced by ≥10%
- ✅ Metadata preserved for all passages