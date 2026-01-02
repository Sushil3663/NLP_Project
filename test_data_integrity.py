"""
Test script for data integrity validation.
Verifies that the validation catches and fixes common data issues.
"""

import json
import os
import tempfile
import shutil
from data_integrity import DataIntegrityValidator

def create_test_data():
    """Create test data with known issues."""
    
    # Create passages with issues
    passages = [
        {
            'passage_id': 'p1',
            'text': 'This is a valid passage with enough text to be meaningful.',
            'title': 'Valid Passage 1',
            'url': 'http://example.com/1',
            'article_id': 'a1'
        },
        {
            'passage_id': 'p2',
            'text': 'Another valid passage with sufficient content for testing purposes.',
            'title': 'Valid Passage 2', 
            'url': 'http://example.com/2',
            'article_id': 'a2'
        },
        {
            'passage_id': 'p1',  # Duplicate ID (issue)
            'text': 'Duplicate passage ID that should be removed.',
            'title': 'Duplicate Passage',
            'url': 'http://example.com/dup',
            'article_id': 'a3'
        },
        {
            'passage_id': 'p3',
            'text': 'Short',  # Too short (issue)
            'title': 'Short Passage',
            'url': 'http://example.com/3',
            'article_id': 'a4'
        },
        {
            'passage_id': 'p4',
            'text': 'This passage will be referenced by queries and should remain.',
            'title': 'Referenced Passage',
            'url': 'http://example.com/4',
            'article_id': 'a5'
        }
    ]
    
    # Create queries with issues
    queries = [
        {
            'query_text': 'What is this about?',
            'positive_passage_id': 'p1',  # Valid reference
            'article_id': 'a1',
            'query': 'What is this about?'
        },
        {
            'query_text': 'Another question?',
            'positive_passage_id': 'p2',  # Valid reference
            'article_id': 'a2',
            'query': 'Another question?'
        },
        {
            'query_text': 'Missing positive?',
            'positive_passage_id': 'p999',  # Missing passage (issue)
            'article_id': 'a999',
            'query': 'Missing positive?'
        },
        {
            'query_text': 'Reference to p4?',
            'positive_passage_id': 'p4',  # Valid reference
            'article_id': 'a5',
            'query': 'Reference to p4?'
        }
    ]
    
    return passages, queries

def test_data_integrity():
    """Test the data integrity validation."""
    
    print("Testing Data Integrity Validation")
    print("=" * 40)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'data')
        output_dir = os.path.join(temp_dir, 'outputs')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create test data with issues
        passages, queries = create_test_data()
        
        # Save test data
        with open(os.path.join(data_dir, 'processed_passages.json'), 'w') as f:
            json.dump(passages, f, indent=2)
        
        with open(os.path.join(data_dir, 'training_queries.json'), 'w') as f:
            json.dump(queries, f, indent=2)
        
        print(f"Created test data:\")\n        print(f\"  Passages: {len(passages)} (with 1 duplicate, 1 too short)\")\n        print(f\"  Queries: {len(queries)} (with 1 missing positive)\")\n        \n        # Run validation\n        validator = DataIntegrityValidator(data_dir, output_dir)\n        result = validator.validate_and_fix(auto_fix=True)\n        \n        # Check results\n        report = result['report']\n        clean_passages = result['clean_passages']\n        clean_queries = result['clean_queries']\n        aligned_splits = result['aligned_splits']\n        \n        print(f\"\\nValidation Results:\")\n        print(f\"  Issues found: {report['summary']['total_issues']}\")\n        print(f\"  Fixes applied: {report['summary']['fixes_applied']}\")\n        print(f\"  Final passages: {len(clean_passages)}\")\n        print(f\"  Final queries: {len(clean_queries)}\")\n        \n        # Verify fixes\n        passage_ids = [p.get('passage_id') for p in clean_passages]\n        assert len(set(passage_ids)) == len(passage_ids), \"Duplicate IDs not removed\"\n        \n        # Check all queries have valid positives\n        passage_id_set = set(passage_ids)\n        for query in clean_queries:\n            pos_id = query.get('positive_passage_id')\n            assert pos_id in passage_id_set, f\"Query references missing passage: {pos_id}\"\n        \n        # Check splits alignment\n        train_passages = aligned_splits['train_passages']\n        val_passages = aligned_splits['val_passages']\n        train_queries = aligned_splits['train_queries']\n        val_queries = aligned_splits['val_queries']\n        \n        train_passage_ids = {p.get('passage_id') for p in train_passages}\n        val_passage_ids = {p.get('passage_id') for p in val_passages}\n        \n        # Verify no overlap\n        assert len(train_passage_ids & val_passage_ids) == 0, \"Train/val overlap found\"\n        \n        # Verify query alignment\n        for query in train_queries:\n            pos_id = query.get('positive_passage_id')\n            assert pos_id in train_passage_ids, f\"Train query references val passage: {pos_id}\"\n        \n        for query in val_queries:\n            pos_id = query.get('positive_passage_id')\n            assert pos_id in val_passage_ids, f\"Val query references train passage: {pos_id}\"\n        \n        print(f\"\\nSplit Validation:\")\n        print(f\"  Train passages: {len(train_passages)}\")\n        print(f\"  Val passages: {len(val_passages)}\")\n        print(f\"  Train queries: {len(train_queries)}\")\n        print(f\"  Val queries: {len(val_queries)}\")\n        print(f\"  ✅ No train/val overlap\")\n        print(f\"  ✅ All queries properly aligned\")\n        \n        # Check if files were saved\n        assert os.path.exists(os.path.join(output_dir, 'data_integrity_report.txt')), \"Report not saved\"\n        assert os.path.exists(os.path.join(data_dir, 'processed_passages_clean.json')), \"Clean passages not saved\"\n        \n        print(f\"\\n✅ All tests passed!\")\n        print(f\"Data integrity validation is working correctly.\")\n\nif __name__ == \"__main__\":\n    test_data_integrity()