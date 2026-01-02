"""
Evaluation utilities for reading comprehension models.
Computes Exact Match (EM) and F1 scores for answer prediction.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Tuple
import torch
from tqdm import tqdm

def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(text))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score between prediction and ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def compute_metrics(predictions: List[str], 
                   ground_truths: List[List[str]]) -> Dict[str, float]:
    """
    Compute EM and F1 scores for a list of predictions.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of lists of ground truth answers (multiple answers per question)
    
    Returns:
        Dictionary with EM and F1 scores
    """
    em_scores = []
    f1_scores = []
    
    for pred, gt_list in zip(predictions, ground_truths):
        # For each prediction, compute scores against all ground truths and take max
        em_score = max([exact_match_score(pred, gt) for gt in gt_list])
        f1_score_val = max([f1_score(pred, gt) for gt in gt_list])
        
        em_scores.append(em_score)
        f1_scores.append(f1_score_val)
    
    return {
        'exact_match': sum(em_scores) / len(em_scores) * 100,
        'f1': sum(f1_scores) / len(f1_scores) * 100,
        'total_questions': len(predictions)
    }

def evaluate_model(model, 
                  data_loader, 
                  tokenizer,
                  device: str = 'cuda',
                  max_answer_length: int = 30) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained reading comprehension model
        data_loader: DataLoader with evaluation data
        tokenizer: Tokenizer for text processing
        device: Device to run evaluation on
        max_answer_length: Maximum allowed answer length
    
    Returns:
        Dictionary with evaluation metrics and detailed results
    """
    model.eval()
    model.to(device)
    
    predictions = []
    ground_truths = []
    confidence_scores = []
    detailed_results = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            question_ids = batch['question_ids'].to(device)
            passage_ids = batch['passage_ids'].to(device)
            
            # Get predictions
            start_positions, end_positions, confidences = model.predict_span(
                question_ids, passage_ids, max_answer_length
            )
            
            # Extract answer texts
            batch_size = question_ids.shape[0]
            
            for i in range(batch_size):
                # Get predicted answer
                start_pos = start_positions[i].item()
                end_pos = end_positions[i].item()
                confidence = confidences[i].item()
                
                passage_text = batch['passage_texts'][i]
                passage_tokens = tokenizer.tokenize(passage_text)
                
                if (start_pos < len(passage_tokens) and 
                    end_pos < len(passage_tokens) and 
                    start_pos <= end_pos):
                    predicted_answer = tokenizer.detokenize(passage_tokens[start_pos:end_pos + 1])
                else:
                    predicted_answer = ""
                
                # Get ground truth answers
                if batch.get('answer_texts'):
                    gt_answers = [batch['answer_texts'][i]] if batch['answer_texts'][i] else [""]
                else:
                    gt_answers = [""]
                
                # Handle impossible questions
                is_impossible = batch.get('is_impossible', [False] * batch_size)[i]
                if is_impossible:
                    gt_answers = [""]
                
                predictions.append(predicted_answer)
                ground_truths.append(gt_answers)
                confidence_scores.append(confidence)
                
                # Store detailed result
                detailed_results.append({
                    'question': batch['question_texts'][i],
                    'passage': passage_text,
                    'predicted_answer': predicted_answer,
                    'ground_truth_answers': gt_answers,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'confidence': confidence,
                    'is_impossible': is_impossible,
                    'exact_match': max([exact_match_score(predicted_answer, gt) for gt in gt_answers]),
                    'f1_score': max([f1_score(predicted_answer, gt) for gt in gt_answers])
                })
    
    # Compute overall metrics
    metrics = compute_metrics(predictions, ground_truths)
    
    # Add confidence statistics
    metrics['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
    metrics['predictions'] = len(predictions)
    
    # Analyze by confidence threshold
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for threshold in confidence_thresholds:
        high_conf_indices = [i for i, conf in enumerate(confidence_scores) if conf >= threshold]
        
        if high_conf_indices:
            high_conf_predictions = [predictions[i] for i in high_conf_indices]
            high_conf_ground_truths = [ground_truths[i] for i in high_conf_indices]
            
            high_conf_metrics = compute_metrics(high_conf_predictions, high_conf_ground_truths)
            metrics[f'em_at_conf_{threshold}'] = high_conf_metrics['exact_match']
            metrics[f'f1_at_conf_{threshold}'] = high_conf_metrics['f1']
            metrics[f'coverage_at_conf_{threshold}'] = len(high_conf_indices) / len(predictions) * 100
    
    return {
        'metrics': metrics,
        'detailed_results': detailed_results
    }

def analyze_errors(detailed_results: List[Dict[str, Any]], 
                  num_examples: int = 10) -> Dict[str, Any]:
    """Analyze common error patterns in predictions."""
    
    # Separate correct and incorrect predictions
    correct_predictions = [r for r in detailed_results if r['exact_match'] == 1.0]
    incorrect_predictions = [r for r in detailed_results if r['exact_match'] == 0.0]
    
    # Sort by confidence
    incorrect_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Analyze error types
    error_types = {
        'no_answer_predicted': 0,
        'wrong_span': 0,
        'partial_match': 0,
        'completely_wrong': 0
    }
    
    for result in incorrect_predictions:
        pred = result['predicted_answer'].strip()
        gt = result['ground_truth_answers'][0].strip()
        
        if not pred:
            error_types['no_answer_predicted'] += 1
        elif result['f1_score'] > 0.5:
            error_types['partial_match'] += 1
        elif any(word in pred.lower() for word in gt.lower().split()):
            error_types['wrong_span'] += 1
        else:
            error_types['completely_wrong'] += 1
    
    # Get examples of each error type
    error_examples = {}
    for error_type in error_types:
        error_examples[error_type] = []
    
    for result in incorrect_predictions[:num_examples * 4]:  # Get more examples to fill categories
        pred = result['predicted_answer'].strip()
        gt = result['ground_truth_answers'][0].strip()
        
        if not pred and len(error_examples['no_answer_predicted']) < num_examples:
            error_examples['no_answer_predicted'].append(result)
        elif (result['f1_score'] > 0.5 and 
              len(error_examples['partial_match']) < num_examples):
            error_examples['partial_match'].append(result)
        elif (any(word in pred.lower() for word in gt.lower().split()) and 
              len(error_examples['wrong_span']) < num_examples):
            error_examples['wrong_span'].append(result)
        elif len(error_examples['completely_wrong']) < num_examples:
            error_examples['completely_wrong'].append(result)
    
    return {
        'total_correct': len(correct_predictions),
        'total_incorrect': len(incorrect_predictions),
        'error_types': error_types,
        'error_examples': error_examples,
        'accuracy': len(correct_predictions) / len(detailed_results) * 100
    }

def create_evaluation_report(metrics: Dict[str, Any], 
                           error_analysis: Dict[str, Any],
                           output_file: str):
    """Create detailed evaluation report."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Reading Comprehension Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Exact Match (EM): {metrics['exact_match']:.2f}%\n")
        f.write(f"F1 Score: {metrics['f1']:.2f}%\n")
        f.write(f"Average Confidence: {metrics['avg_confidence']:.3f}\n")
        f.write(f"Total Questions: {metrics['total_questions']}\n\n")
        
        # Confidence-based metrics
        f.write("CONFIDENCE-BASED METRICS\n")
        f.write("-" * 25 + "\n")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if f'em_at_conf_{threshold}' in metrics:
                f.write(f"At confidence ≥ {threshold}:\n")
                f.write(f"  EM: {metrics[f'em_at_conf_{threshold}']:.2f}%\n")
                f.write(f"  F1: {metrics[f'f1_at_conf_{threshold}']:.2f}%\n")
                f.write(f"  Coverage: {metrics[f'coverage_at_conf_{threshold}']:.2f}%\n\n")
        
        # Error analysis
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total Correct: {error_analysis['total_correct']}\n")
        f.write(f"Total Incorrect: {error_analysis['total_incorrect']}\n")
        f.write(f"Accuracy: {error_analysis['accuracy']:.2f}%\n\n")
        
        f.write("Error Types:\n")
        for error_type, count in error_analysis['error_types'].items():
            percentage = count / error_analysis['total_incorrect'] * 100 if error_analysis['total_incorrect'] > 0 else 0
            f.write(f"  {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n")
        
        # Error examples
        f.write("ERROR EXAMPLES\n")
        f.write("-" * 15 + "\n")
        for error_type, examples in error_analysis['error_examples'].items():
            if examples:
                f.write(f"\n{error_type.replace('_', ' ').title()}:\n")
                for i, example in enumerate(examples[:3]):  # Show top 3 examples
                    f.write(f"  Example {i+1}:\n")
                    f.write(f"    Question: {example['question'][:100]}...\n")
                    f.write(f"    Predicted: '{example['predicted_answer']}'\n")
                    f.write(f"    Ground Truth: '{example['ground_truth_answers'][0]}'\n")
                    f.write(f"    Confidence: {example['confidence']:.3f}\n")
                    f.write(f"    F1: {example['f1_score']:.3f}\n\n")

if __name__ == "__main__":
    # Test evaluation functions
    
    # Sample predictions and ground truths
    predictions = [
        "Barack Obama",
        "artificial intelligence",
        "2021",
        "",
        "machine learning algorithms"
    ]
    
    ground_truths = [
        ["Barack Obama", "Obama"],
        ["AI", "artificial intelligence"],
        ["2021", "twenty twenty-one"],
        ["no answer"],
        ["deep learning", "neural networks"]
    ]
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)
    
    print("Test Evaluation Results:")
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    print(f"Total Questions: {metrics['total_questions']}")
    
    # Test individual scores
    print("\nIndividual Scores:")
    for i, (pred, gt_list) in enumerate(zip(predictions, ground_truths)):
        em = max([exact_match_score(pred, gt) for gt in gt_list])
        f1 = max([f1_score(pred, gt) for gt in gt_list])
        print(f"  Q{i+1}: EM={em:.1f}, F1={f1:.3f} | Pred: '{pred}' | GT: {gt_list}")
    
    # Test normalization
    print("\nNormalization Test:")
    test_pairs = [
        ("The United States", "united states"),
        ("A.I.", "AI"),
        ("twenty-one", "21"),
    ]
    
    for pred, gt in test_pairs:
        em = exact_match_score(pred, gt)
        f1 = f1_score(pred, gt)
        print(f"  '{pred}' vs '{gt}': EM={em:.1f}, F1={f1:.3f}")
        print(f"    Normalized: '{normalize_answer(pred)}' vs '{normalize_answer(gt)}'")