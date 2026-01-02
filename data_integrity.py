"""
Data Integrity Validation Module for LSTM News QA System.
Ensures consistency across passages, queries, splits, and indexes.
"""

import json
import os
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import logging

class DataIntegrityValidator:
    """Validates data consistency across the entire pipeline."""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.report = {
            'timestamp': None,
            'passages': {},
            'queries': {},
            'splits': {},
            'integrity_issues': [],
            'fixes_applied': [],
            'summary': {}
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load passages and queries."""
        passages_file = os.path.join(self.data_dir, 'processed_passages.json')
        queries_file = os.path.join(self.data_dir, 'training_queries.json')
        
        passages = []
        queries = []
        
        if os.path.exists(passages_file):
            with open(passages_file, 'r', encoding='utf-8') as f:
                passages = json.load(f)
        
        if os.path.exists(queries_file):
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
        
        return passages, queries
    
    def validate_passage_consistency(self, passages: List[Dict]) -> Dict[str, Any]:
        """Validate passage data consistency."""
        issues = []
        
        # Check unique passage IDs
        passage_ids = [p.get('passage_id') for p in passages]
        id_counts = Counter(passage_ids)
        duplicates = {pid: count for pid, count in id_counts.items() if count > 1}
        
        if duplicates:
            issues.append(f"Duplicate passage IDs found: {duplicates}")
        
        # Check required fields
        required_fields = ['passage_id', 'text', 'title', 'url']
        missing_fields = defaultdict(int)
        
        for passage in passages:
            for field in required_fields:
                if not passage.get(field):
                    missing_fields[field] += 1
        
        if missing_fields:
            issues.append(f"Missing required fields: {dict(missing_fields)}")
        
        # Check text quality
        empty_text = sum(1 for p in passages if not p.get('text', '').strip())
        short_text = sum(1 for p in passages if len(p.get('text', '').split()) < 10)
        
        if empty_text > 0:
            issues.append(f"Passages with empty text: {empty_text}")
        if short_text > 0:
            issues.append(f"Passages with <10 words: {short_text}")
        
        return {
            'total_passages': len(passages),
            'unique_ids': len(set(passage_ids)),
            'duplicate_ids': len(duplicates),
            'issues': issues
        }
    
    def validate_query_consistency(self, queries: List[Dict], passages: List[Dict]) -> Dict[str, Any]:
        """Validate query data consistency."""
        issues = []
        
        # Create passage ID lookup
        passage_ids = {p.get('passage_id') for p in passages}
        
        # Check query structure
        missing_positives = []
        invalid_queries = []
        
        for i, query in enumerate(queries):
            # Check required fields
            if not query.get('query_text'):
                invalid_queries.append(i)
                continue
            
            # Check positive passage exists
            pos_id = query.get('positive_passage_id')
            if pos_id and pos_id not in passage_ids:
                missing_positives.append({
                    'query_idx': i,
                    'query_text': query.get('query_text', '')[:50],
                    'missing_passage_id': pos_id
                })
        
        if missing_positives:
            issues.append(f"Queries with missing positive passages: {len(missing_positives)}")
        
        if invalid_queries:
            issues.append(f"Invalid queries (missing text): {len(invalid_queries)}")
        
        return {
            'total_queries': len(queries),
            'missing_positives': missing_positives,
            'invalid_queries': invalid_queries,
            'issues': issues
        }
    
    def validate_splits_consistency(self, passages: List[Dict], queries: List[Dict], 
                                  train_split: float = 0.9) -> Dict[str, Any]:
        """Validate train/validation split consistency."""
        issues = []
        
        # Create splits
        split_idx = int(len(passages) * train_split)
        train_passages = passages[:split_idx]
        val_passages = passages[split_idx:]
        
        train_passage_ids = {p.get('passage_id') for p in train_passages}
        val_passage_ids = {p.get('passage_id') for p in val_passages}
        
        # Check for overlap (should be none)
        overlap = train_passage_ids & val_passage_ids
        if overlap:
            issues.append(f"Train/val passage overlap: {len(overlap)} passages")
        
        # Check query alignment
        train_queries = []
        val_queries = []
        orphaned_queries = []
        
        for query in queries:
            pos_id = query.get('positive_passage_id')
            if pos_id in train_passage_ids:
                train_queries.append(query)
            elif pos_id in val_passage_ids:
                val_queries.append(query)
            else:
                orphaned_queries.append(query)
        
        if orphaned_queries:
            issues.append(f"Orphaned queries (no matching passage in splits): {len(orphaned_queries)}")
        
        return {
            'train_passages': len(train_passages),
            'val_passages': len(val_passages),
            'train_queries': len(train_queries),
            'val_queries': len(val_queries),
            'orphaned_queries': len(orphaned_queries),
            'overlap_passages': len(overlap),
            'issues': issues,
            'splits': {
                'train_passages': train_passages,
                'val_passages': val_passages,
                'train_queries': train_queries,
                'val_queries': val_queries
            }
        }
    
    def fix_data_issues(self, passages: List[Dict], queries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Apply automatic fixes to data issues."""
        fixes = []
        
        # Fix 1: Remove duplicate passages
        seen_ids = set()
        clean_passages = []
        
        for passage in passages:
            pid = passage.get('passage_id')
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                clean_passages.append(passage)
            else:
                fixes.append(f"Removed duplicate passage: {pid}")
        
        # Fix 2: Remove queries with missing positives
        passage_ids = {p.get('passage_id') for p in clean_passages}
        clean_queries = []
        
        for query in queries:
            pos_id = query.get('positive_passage_id')
            if pos_id in passage_ids:
                clean_queries.append(query)
            else:
                fixes.append(f"Removed query with missing positive: {query.get('query_text', '')[:50]}")
        
        # Fix 3: Ensure minimum text length
        filtered_passages = []
        for passage in clean_passages:
            text = passage.get('text', '').strip()
            if len(text.split()) >= 10:  # Minimum 10 words
                filtered_passages.append(passage)
            else:
                fixes.append(f"Removed short passage: {passage.get('passage_id')}")
        
        self.report['fixes_applied'] = fixes
        return filtered_passages, clean_queries
    
    def create_aligned_splits(self, passages: List[Dict], queries: List[Dict], 
                            train_split: float = 0.9) -> Dict[str, List[Dict]]:
        """Create properly aligned train/validation splits."""
        
        # Strategy: Split passages first, then align queries
        split_idx = int(len(passages) * train_split)
        train_passages = passages[:split_idx]
        val_passages = passages[split_idx:]
        
        train_passage_ids = {p.get('passage_id') for p in train_passages}
        val_passage_ids = {p.get('passage_id') for p in val_passages}
        
        # Align queries with splits
        train_queries = []
        val_queries = []
        
        for query in queries:
            pos_id = query.get('positive_passage_id')
            if pos_id in train_passage_ids:
                train_queries.append(query)
            elif pos_id in val_passage_ids:
                val_queries.append(query)
        
        return {
            'train_passages': train_passages,
            'val_passages': val_passages,
            'train_queries': train_queries,
            'val_queries': val_queries,
            'all_passages': passages,  # For building complete index
            'all_queries': queries
        }
    
    def validate_and_fix(self, auto_fix: bool = True) -> Dict[str, Any]:
        """Run complete validation and optionally fix issues."""
        from datetime import datetime
        
        self.report['timestamp'] = datetime.now().isoformat()
        
        print("Loading data...")
        passages, queries = self.load_data()
        
        if not passages:
            raise ValueError("No passages found. Run Week 1 data processing first.")
        
        print(f"Loaded {len(passages)} passages, {len(queries)} queries")
        
        # Validate passages
        print("Validating passages...")
        passage_validation = self.validate_passage_consistency(passages)
        self.report['passages'] = passage_validation
        
        # Validate queries
        print("Validating queries...")
        query_validation = self.validate_query_consistency(queries, passages)
        self.report['queries'] = query_validation
        
        # Apply fixes if requested
        if auto_fix and (passage_validation['issues'] or query_validation['issues']):
            print("Applying automatic fixes...")
            passages, queries = self.fix_data_issues(passages, queries)
            
            # Re-validate after fixes
            passage_validation = self.validate_passage_consistency(passages)
            query_validation = self.validate_query_consistency(queries, passages)
        
        # Validate splits
        print("Validating splits...")
        split_validation = self.validate_splits_consistency(passages, queries)
        self.report['splits'] = split_validation
        
        # Create aligned splits
        print("Creating aligned splits...")
        aligned_splits = self.create_aligned_splits(passages, queries)
        
        # Save cleaned data
        if auto_fix:
            self.save_cleaned_data(passages, queries, aligned_splits)
        
        # Generate summary
        self.report['summary'] = {
            'total_issues': len(passage_validation['issues']) + len(query_validation['issues']) + len(split_validation['issues']),
            'fixes_applied': len(self.report.get('fixes_applied', [])),
            'final_passages': len(passages),
            'final_queries': len(queries),
            'train_passages': len(aligned_splits['train_passages']),
            'val_passages': len(aligned_splits['val_passages']),
            'train_queries': len(aligned_splits['train_queries']),
            'val_queries': len(aligned_splits['val_queries'])
        }
        
        # Save report
        self.save_report()
        
        return {
            'report': self.report,
            'aligned_splits': aligned_splits,
            'clean_passages': passages,
            'clean_queries': queries
        }
    
    def save_cleaned_data(self, passages: List[Dict], queries: List[Dict], 
                         aligned_splits: Dict[str, List[Dict]]):
        """Save cleaned and aligned data."""
        
        # Save cleaned passages and queries
        with open(os.path.join(self.data_dir, 'processed_passages_clean.json'), 'w', encoding='utf-8') as f:
            json.dump(passages, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(self.data_dir, 'training_queries_clean.json'), 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        
        # Save aligned splits
        splits_dir = os.path.join(self.data_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        for split_name, split_data in aligned_splits.items():
            if split_name != 'all_passages' and split_name != 'all_queries':
                with open(os.path.join(splits_dir, f'{split_name}.json'), 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved cleaned data to {self.data_dir}")
    
    def save_report(self):
        """Save integrity validation report."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # JSON report
        report_file = os.path.join(self.output_dir, 'data_integrity_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        # Human-readable report
        text_report = os.path.join(self.output_dir, 'data_integrity_report.txt')
        with open(text_report, 'w', encoding='utf-8') as f:
            f.write("LSTM News QA System - Data Integrity Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {self.report['timestamp']}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = self.report['summary']
            f.write(f"Total issues found: {summary['total_issues']}\n")
            f.write(f"Fixes applied: {summary['fixes_applied']}\n")
            f.write(f"Final passages: {summary['final_passages']}\n")
            f.write(f"Final queries: {summary['final_queries']}\n")
            f.write(f"Train passages: {summary['train_passages']}\n")
            f.write(f"Val passages: {summary['val_passages']}\n")
            f.write(f"Train queries: {summary['train_queries']}\n")
            f.write(f"Val queries: {summary['val_queries']}\n\n")
            
            f.write("PASSAGE VALIDATION\n")
            f.write("-" * 20 + "\n")
            for issue in self.report['passages']['issues']:
                f.write(f"❌ {issue}\n")
            if not self.report['passages']['issues']:
                f.write("✅ No passage issues found\n")
            f.write("\n")
            
            f.write("QUERY VALIDATION\n")
            f.write("-" * 20 + "\n")
            for issue in self.report['queries']['issues']:
                f.write(f"❌ {issue}\n")
            if not self.report['queries']['issues']:
                f.write("✅ No query issues found\n")
            f.write("\n")
            
            f.write("SPLIT VALIDATION\n")
            f.write("-" * 20 + "\n")
            for issue in self.report['splits']['issues']:
                f.write(f"❌ {issue}\n")
            if not self.report['splits']['issues']:
                f.write("✅ No split issues found\n")
            f.write("\n")
            
            if self.report.get('fixes_applied'):
                f.write("FIXES APPLIED\n")
                f.write("-" * 20 + "\n")
                for fix in self.report['fixes_applied']:
                    f.write(f"🔧 {fix}\n")
        
        print(f"Saved integrity report to {text_report}")

def main():
    """Run data integrity validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate data integrity')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no_fix', action='store_true', help='Skip automatic fixes')
    
    args = parser.parse_args()
    
    validator = DataIntegrityValidator(args.data_dir, args.output_dir)
    result = validator.validate_and_fix(auto_fix=not args.no_fix)
    
    print("\nData integrity validation completed!")
    print(f"Issues found: {result['report']['summary']['total_issues']}")
    print(f"Fixes applied: {result['report']['summary']['fixes_applied']}")

if __name__ == "__main__":
    main()