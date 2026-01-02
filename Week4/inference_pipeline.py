"""
Main inference pipeline for the LSTM-based News QA System.
Orchestrates the complete end-to-end question answering process.
"""

import os
import sys
import json
import torch
import pickle
from typing import Dict, Any, List, Optional
import argparse
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Week2.retriever_model import DualEncoder
from Week2.embeddings import EmbeddingManager
from Week2.indexing import FAISSIndex
from Week3.reader_model import LSTMReader
from Week3.squad_data import SimpleTokenizer
from question_processor import QuestionProcessor
from reranker import PassageReranker, create_simple_reranker
from answer_aggregator import AnswerAggregator
from citation_formatter import CitationFormatter

class NewsQASystem:
    """Complete News Question Answering System."""
    
    def __init__(self, 
                 model_dir: str,
                 device: str = 'auto',
                 confidence_threshold: float = 0.3):
        
        self.model_dir = model_dir
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.question_processor = QuestionProcessor()
        self.answer_aggregator = AnswerAggregator(confidence_threshold)
        self.citation_formatter = CitationFormatter()
        
        # Model components (loaded lazily)
        self.embedding_manager = None
        self.retriever_model = None
        self.faiss_index = None
        self.reader_model = None
        self.reader_tokenizer = None
        self.reranker = None
        
        # Load models
        self._load_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_models(self):
        """Load all required models and components."""
        print("Loading QA system components...")
        
        try:
            # Load embedding manager
            self._load_embedding_manager()
            
            # Load retriever model and FAISS index
            self._load_retriever_components()
            
            # Load reader model
            self._load_reader_components()
            
            # Create reranker
            self._create_reranker()
            
            print("All components loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure all models are trained and saved properly.")
            raise
    
    def _load_embedding_manager(self):
        """Load embedding manager."""
        embedding_file = os.path.join(self.model_dir, 'embeddings.pkl')
        
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding manager not found: {embedding_file}")
        
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.load(embedding_file)
        print(f"Loaded embedding manager with {self.embedding_manager.vocab_size} words")
    
    def _load_retriever_components(self):
        """Load retriever model and FAISS index."""
        # Load retriever model
        retriever_file = os.path.join(self.model_dir, 'best_model.pth')
        if not os.path.exists(retriever_file):
            retriever_file = os.path.join(self.model_dir, 'retriever_model.pth')
        
        if os.path.exists(retriever_file):
            checkpoint = torch.load(retriever_file, map_location=self.device)
            
            # Create retriever model
            pretrained_embeddings = torch.from_numpy(self.embedding_manager.embeddings).float()
            self.retriever_model = DualEncoder(
                vocab_size=self.embedding_manager.vocab_size,
                embedding_dim=self.embedding_manager.embedding_dim,
                hidden_dim=128,  # Use lower dimensions
                output_dim=128,  # Use lower dimensions
                pretrained_embeddings=pretrained_embeddings
            )
            
            self.retriever_model.load_state_dict(checkpoint['model_state_dict'])
            self.retriever_model.to(self.device)
            self.retriever_model.eval()
            
            print("Loaded retriever model")
        else:
            print("Warning: Retriever model not found, using embedding similarity")
        
        # Load FAISS index
        index_file = os.path.join(self.model_dir, 'faiss_index')
        
        if os.path.exists(index_file + '.index'):
            self.faiss_index = FAISSIndex(embedding_dim=128)  # Use lower dimension
            success = self.faiss_index.load(index_file)
            
            if success:
                print(f"Loaded FAISS index with {self.faiss_index.index.ntotal} passages")
            else:
                print("Warning: Failed to load FAISS index")
        else:
            print("Warning: FAISS index not found")
    
    def _load_reader_components(self):
        """Load reader model and tokenizer."""
        # Load reader tokenizer
        tokenizer_file = os.path.join(self.model_dir, 'reader_tokenizer.pkl')
        
        if os.path.exists(tokenizer_file):
            with open(tokenizer_file, 'rb') as f:
                tokenizer_data = pickle.load(f)
            
            self.reader_tokenizer = SimpleTokenizer()
            self.reader_tokenizer.word_to_idx = tokenizer_data['word_to_idx']
            self.reader_tokenizer.idx_to_word = tokenizer_data['idx_to_word']
            self.reader_tokenizer.vocab_size = tokenizer_data['vocab_size']
            
            print(f"Loaded reader tokenizer with {self.reader_tokenizer.vocab_size} words")
        else:
            print("Warning: Reader tokenizer not found")
            return
        
        # Load reader model
        reader_files = [
            os.path.join(self.model_dir, 'best_reader_news.pth'),
            os.path.join(self.model_dir, 'best_reader_squad.pth'),
            os.path.join(self.model_dir, 'reader_model.pth')
        ]
        
        reader_file = None
        for file_path in reader_files:
            if os.path.exists(file_path):
                reader_file = file_path
                break
        
        if reader_file:
            checkpoint = torch.load(reader_file, map_location=self.device)
            
            self.reader_model = LSTMReader(
                vocab_size=self.reader_tokenizer.vocab_size,
                embedding_dim=300,
                hidden_dim=256,
                num_layers=2,
                dropout=0.1
            )
            
            self.reader_model.load_state_dict(checkpoint['model_state_dict'])
            self.reader_model.to(self.device)
            self.reader_model.eval()
            
            print(f"Loaded reader model from {reader_file}")
        else:
            print("Warning: Reader model not found")
    
    def _create_reranker(self):
        """Create reranker component."""
        if self.embedding_manager:
            self.reranker = create_simple_reranker(self.embedding_manager, self.device)
            print("Created reranker component")
        else:
            print("Warning: Cannot create reranker without embedding manager")
    
    def retrieve_passages(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve relevant passages for the question."""
        if not self.faiss_index:
            print("Warning: No FAISS index available")
            return []
        
        try:
            if self.retriever_model:
                # Use trained retriever model
                question_ids = self.embedding_manager.text_to_indices(question, 64)
                question_tensor = torch.tensor([question_ids], dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    query_embedding = self.retriever_model.encode_query(question_tensor)
                    query_embedding = query_embedding.cpu().numpy()
            else:
                # Fallback to simple embedding average
                question_words = question.lower().split()
                embeddings = []
                
                for word in question_words:
                    if word in self.embedding_manager.word_to_idx:
                        idx = self.embedding_manager.word_to_idx[word]
                        embeddings.append(self.embedding_manager.embeddings[idx])
                
                if embeddings:
                    query_embedding = np.mean(embeddings, axis=0, keepdims=True)
                else:
                    # Random embedding as last resort
                    query_embedding = np.random.randn(1, self.embedding_manager.embedding_dim)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Get passage metadata
            retrieved_passages = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(self.faiss_index.passage_metadata):
                    passage = self.faiss_index.passage_metadata[idx].copy()
                    passage['retrieval_score'] = float(score)
                    passage['retrieval_rank'] = i + 1
                    retrieved_passages.append(passage)
            
            return retrieved_passages
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def rerank_passages(self, question: str, passages: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank retrieved passages."""
        if not self.reranker or not passages:
            return passages[:top_k]
        
        try:
            reranked = self.reranker.rerank_passages(question, passages, top_k)
            return reranked
        except Exception as e:
            print(f"Error during re-ranking: {e}")
            return passages[:top_k]
    
    def extract_answers(self, question: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract answer candidates from passages."""
        if not self.reader_model or not self.reader_tokenizer or not passages:
            return []
        
        candidates = []
        
        try:
            for passage in passages:
                passage_text = passage.get('text', '')
                if not passage_text:
                    continue
                
                # Prepare input
                question_ids = self.reader_tokenizer.text_to_indices(question, 64)
                passage_ids = self.reader_tokenizer.text_to_indices(passage_text, 512)
                
                question_tensor = torch.tensor([question_ids], dtype=torch.long).to(self.device)
                passage_tensor = torch.tensor([passage_ids], dtype=torch.long).to(self.device)
                
                # Get answer span
                with torch.no_grad():
                    start_pos, end_pos, confidence = self.reader_model.predict_span(
                        question_tensor, passage_tensor, max_answer_length=30
                    )
                
                start_pos = start_pos[0].item()
                end_pos = end_pos[0].item()
                confidence = confidence[0].item()
                
                # Extract answer text
                passage_tokens = self.reader_tokenizer.tokenize(passage_text)
                
                if (0 <= start_pos < len(passage_tokens) and 
                    0 <= end_pos < len(passage_tokens) and 
                    start_pos <= end_pos):
                    
                    answer_tokens = passage_tokens[start_pos:end_pos + 1]
                    answer_text = self.reader_tokenizer.detokenize(answer_tokens)
                    
                    if answer_text.strip():
                        candidate = {
                            'answer_text': answer_text.strip(),
                            'confidence': confidence,
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'passage_text': passage_text,
                            'passage_score': passage.get('rerank_score', passage.get('retrieval_score', 0.5)),
                            'title': passage.get('title', ''),
                            'url': passage.get('url', ''),
                            'publisher': passage.get('publisher', ''),
                            'date': passage.get('date', ''),
                            'author': passage.get('author', '')
                        }
                        candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"Error during answer extraction: {e}")
            return []
    
    def answer_question(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Answer a question using the complete pipeline.
        
        Args:
            question: Input question
            verbose: Whether to print intermediate steps
        
        Returns:
            Formatted answer with citations
        """
        if verbose:
            print(f"Processing question: {question}")
        
        # Step 1: Preprocess question
        question_analysis = self.question_processor.process_question(question)
        
        if verbose:
            print(f"Question type: {question_analysis['question_type']}")
            print(f"Keywords: {', '.join(question_analysis['keywords'][:5])}")
        
        # Step 2: Retrieve passages
        retrieved_passages = self.retrieve_passages(question, top_k=20)
        
        if verbose:
            print(f"Retrieved {len(retrieved_passages)} passages")
        
        if not retrieved_passages:
            return self.citation_formatter.format_final_answer({
                'is_fallback': True,
                'message': f'No relevant passages found for: "{question}"',
                'top_sources': []
            })
        
        # Step 3: Re-rank passages
        reranked_passages = self.rerank_passages(question, retrieved_passages, top_k=5)
        
        if verbose:
            print(f"Re-ranked to top {len(reranked_passages)} passages")
        
        # Step 4: Extract answer candidates
        answer_candidates = self.extract_answers(question, reranked_passages)
        
        if verbose:
            print(f"Extracted {len(answer_candidates)} answer candidates")
        
        # Step 5: Aggregate answers
        aggregated_answers = self.answer_aggregator.aggregate_answers(
            answer_candidates, question_analysis, max_answers=3
        )
        
        # Step 6: Format final answer
        if aggregated_answers and self.answer_aggregator.should_return_answer(aggregated_answers[0]):
            best_answer = aggregated_answers[0]
            formatted_answer = self.citation_formatter.format_final_answer(best_answer)
            
            if verbose:
                print(f"Answer confidence: {best_answer['confidence']:.3f}")
            
            return formatted_answer
        else:
            # Return fallback response
            fallback_data = self.answer_aggregator.create_fallback_response(
                reranked_passages, question
            )
            return self.citation_formatter.format_final_answer(fallback_data)
    
    def batch_answer_questions(self, questions: List[str], verbose: bool = False) -> List[Dict[str, Any]]:
        """Answer multiple questions."""
        results = []
        
        for i, question in enumerate(questions, 1):
            if verbose:
                print(f"\n--- Question {i}/{len(questions)} ---")
            
            try:
                answer = self.answer_question(question, verbose=verbose)
                results.append({
                    'question': question,
                    'answer': answer,
                    'success': True
                })
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                results.append({
                    'question': question,
                    'answer': {'answer': 'error', 'message': str(e)},
                    'success': False
                })
        
        return results

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='LSTM News QA System')
    parser.add_argument('question', nargs='?', help='Question to answer')
    parser.add_argument('--model_dir', type=str, default='../models', 
                       help='Directory containing trained models')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                       help='Confidence threshold for returning answers')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'], help='Computation device')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--batch_file', type=str,
                       help='File containing questions to answer (one per line)')
    parser.add_argument('--output_file', type=str,
                       help='Output file for batch results')
    
    args = parser.parse_args()
    
    # Adjust model directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, 'models')
    
    if args.model_dir != '../models':
        model_dir = args.model_dir
    
    print("LSTM News QA System")
    print("=" * 30)
    print(f"Model directory: {model_dir}")
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print()
    
    try:
        # Initialize QA system
        qa_system = NewsQASystem(
            model_dir=model_dir,
            device=args.device,
            confidence_threshold=args.confidence_threshold
        )
        
        if args.batch_file:
            # Batch processing
            print(f"Processing questions from: {args.batch_file}")
            
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            
            results = qa_system.batch_answer_questions(questions, verbose=args.verbose)
            
            # Save results
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {args.output_file}")
            
            # Print summary
            successful = sum(1 for r in results if r['success'])
            print(f"\nProcessed {len(questions)} questions")
            print(f"Successful: {successful}")
            print(f"Failed: {len(questions) - successful}")
        
        elif args.interactive:
            # Interactive mode
            print("Interactive mode. Type 'quit' to exit.")
            
            while True:
                try:
                    question = input("\nQuestion: ").strip()
                    
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not question:
                        continue
                    
                    print("\nProcessing...")
                    answer = qa_system.answer_question(question, verbose=args.verbose)
                    display_text = qa_system.citation_formatter.format_display_text(answer)
                    
                    print("\n" + "=" * 50)
                    print(display_text)
                    print("=" * 50)
                
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        elif args.question:
            # Single question
            print(f"Question: {args.question}")
            print("\nProcessing...")
            
            answer = qa_system.answer_question(args.question, verbose=args.verbose)
            display_text = qa_system.citation_formatter.format_display_text(answer)
            
            print("\n" + "=" * 50)
            print(display_text)
            print("=" * 50)
        
        else:
            # Default demo questions
            demo_questions = [
                "What is artificial intelligence?",
                "Who invented the telephone?",
                "What are the latest developments in quantum computing?",
                "How does machine learning work?",
                "What is the impact of climate change?"
            ]
            
            print("Running demo with sample questions...")
            results = qa_system.batch_answer_questions(demo_questions, verbose=args.verbose)
            
            print("\n" + "=" * 60)
            print("DEMO RESULTS")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Question: {result['question']}")
                if result['success']:
                    display_text = qa_system.citation_formatter.format_display_text(result['answer'])
                    print(display_text)
                else:
                    print(f"Error: {result['answer']['message']}")
                print("-" * 40)
    
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        print("\nPlease ensure:")
        print("1. All models are trained (run Week 1-3 training scripts)")
        print("2. Model files are in the correct directory")
        print("3. Required dependencies are installed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())