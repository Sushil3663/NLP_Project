"""
Main data processing pipeline for Week 1.
Processes raw news articles into clean, chunked, and deduplicated passages.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import spacy
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import clean_article, TextCleaner
from chunking import PassageChunker, create_training_queries
from deduplication import deduplicate_passages

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    sys.exit(1)

class DataProcessor:
    """Main data processing pipeline."""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.text_cleaner = TextCleaner()
        self.chunker = PassageChunker(min_tokens=200, max_tokens=400, overlap_tokens=50)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)
    
    def load_articles(self) -> List[Dict[str, Any]]:
        """Load articles from input JSON file."""
        print(f"Loading articles from {self.input_file}...")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            print(f"Loaded {len(articles)} articles")
            return articles
        
        except Exception as e:
            print(f"Error loading articles: {e}")
            return []
    
    def clean_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and normalize article text."""
        print("Cleaning articles...")
        
        cleaned_articles = []
        for article in tqdm(articles, desc="Cleaning"):
            try:
                cleaned = clean_article(article)
                if cleaned.get('text') and len(cleaned['text'].strip()) > 100:
                    cleaned_articles.append(cleaned)
            except Exception as e:
                print(f"Error cleaning article {article.get('url', 'unknown')}: {e}")
        
        print(f"Cleaned {len(cleaned_articles)} articles (removed {len(articles) - len(cleaned_articles)})")
        return cleaned_articles
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using spaCy."""
        try:
            doc = nlp(text[:1000])  # Limit text length for efficiency
            entities = []
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_
                    })
            
            return entities
        except Exception:
            return []
    
    def chunk_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk articles into passages."""
        print("Chunking articles into passages...")
        
        all_passages = []
        for article in tqdm(articles, desc="Chunking"):
            try:
                passages = self.chunker.chunk_article(article)
                
                # Add entity extraction to each passage
                for passage in passages:
                    passage['entities'] = self.extract_entities(passage['text'])
                
                all_passages.extend(passages)
            
            except Exception as e:
                print(f"Error chunking article {article.get('url', 'unknown')}: {e}")
        
        print(f"Created {len(all_passages)} passages")
        return all_passages
    
    def deduplicate_passages(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove near-duplicate passages."""
        print("Deduplicating passages...")
        
        deduplicated, stats = deduplicate_passages(passages, threshold=0.8)
        
        print(f"Deduplication results:")
        print(f"  Original: {stats['original_count']} passages")
        print(f"  After deduplication: {stats['deduplicated_count']} passages")
        print(f"  Removed: {stats['removed_count']} passages ({stats['reduction_percentage']:.1f}%)")
        print(f"  Duplicate groups: {stats['duplicate_groups']}")
        
        return deduplicated, stats
    
    def save_results(self, passages: List[Dict[str, Any]], stats: Dict[str, Any]):
        """Save processed passages and statistics."""
        # Save passages
        passages_file = os.path.join(self.output_dir, 'data', 'processed_passages.json')
        with open(passages_file, 'w', encoding='utf-8') as f:
            json.dump(passages, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(passages)} passages to {passages_file}")
        
        # Create training queries
        queries = create_training_queries(passages)
        queries_file = os.path.join(self.output_dir, 'data', 'training_queries.json')
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        print(f"Created {len(queries)} training queries")
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'outputs', 'week1_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        # Generate report
        self.generate_report(passages, stats)
    
    def generate_report(self, passages: List[Dict[str, Any]], dedup_stats: Dict[str, Any]):
        """Generate detailed processing report."""
        report_file = os.path.join(self.output_dir, 'outputs', 'week1_report.txt')
        
        # Calculate statistics
        total_passages = len(passages)
        total_tokens = sum(p.get('token_count', 0) for p in passages)
        avg_tokens = total_tokens / total_passages if total_passages > 0 else 0
        
        publishers = {}
        dates = {}
        entities_count = 0
        
        for passage in passages:
            # Publisher stats
            pub = passage.get('publisher', 'unknown')
            publishers[pub] = publishers.get(pub, 0) + 1
            
            # Date stats
            date = passage.get('date', 'unknown')[:10] if passage.get('date') else 'unknown'
            dates[date] = dates.get(date, 0) + 1
            
            # Entity stats
            entities_count += len(passage.get('entities', []))
        
        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("LSTM News QA System - Week 1 Data Processing Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PROCESSING SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total passages created: {total_passages:,}\n")
            f.write(f"Total tokens: {total_tokens:,}\n")
            f.write(f"Average tokens per passage: {avg_tokens:.1f}\n")
            f.write(f"Total entities extracted: {entities_count:,}\n\n")
            
            f.write("DEDUPLICATION RESULTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Original passages: {dedup_stats['original_count']:,}\n")
            f.write(f"After deduplication: {dedup_stats['deduplicated_count']:,}\n")
            f.write(f"Removed duplicates: {dedup_stats['removed_count']:,}\n")
            f.write(f"Reduction percentage: {dedup_stats['reduction_percentage']:.1f}%\n")
            f.write(f"Duplicate groups found: {dedup_stats['duplicate_groups']:,}\n\n")
            
            f.write("PUBLISHER DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            for pub, count in sorted(publishers.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"{pub}: {count:,} passages\n")
            
            f.write(f"\nSUCCESS CRITERIA CHECK\n")
            f.write("-" * 20 + "\n")
            f.write(f"At least 5,000 passages: {total_passages >= 5000} ({total_passages:,})\n")
            f.write(f"Duplicates reduced by ≥10%: {dedup_stats['reduction_percentage'] >= 10.0} ({dedup_stats['reduction_percentage']:.1f}%)\n")
            f.write(f"Metadata preserved: {all(p.get('url') and p.get('title') for p in passages[:100])}\n")
        
        print(f"Generated report: {report_file}")
    
    def process(self):
        """Run the complete data processing pipeline."""
        print("Starting Week 1 data processing pipeline...")
        print("=" * 50)
        
        # Load articles
        articles = self.load_articles()
        if not articles:
            print("No articles to process!")
            return
        
        # Clean articles
        cleaned_articles = self.clean_articles(articles)
        
        # Chunk into passages
        passages = self.chunk_articles(cleaned_articles)
        
        # Deduplicate
        deduplicated_passages, dedup_stats = self.deduplicate_passages(passages)
        
        # Add processing metadata
        processing_stats = {
            'processing_date': datetime.now().isoformat(),
            'original_articles': len(articles),
            'cleaned_articles': len(cleaned_articles),
            'total_passages': len(passages),
            'final_passages': len(deduplicated_passages),
            'deduplication': dedup_stats
        }
        
        # Save results
        self.save_results(deduplicated_passages, processing_stats)
        
        print("\n" + "=" * 50)
        print("Week 1 processing completed successfully!")
        print(f"Final output: {len(deduplicated_passages)} clean passages")

def main():
    """Main entry point."""
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_file = os.path.join(project_root, 'cleaned_data.json')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please ensure cleaned_data.json is in the project root directory.")
        return
    
    # Process data
    processor = DataProcessor(input_file, project_root)
    processor.process()

if __name__ == "__main__":
    main()