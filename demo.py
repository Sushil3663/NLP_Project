"""
Demo script for the LSTM-based News QA System.
Provides an easy interface to test the complete question answering pipeline.
"""

import os
import sys
import argparse
from datetime import datetime

# Add Week4 to path
current_dir = os.path.dirname(os.path.abspath(__file__))
week4_dir = os.path.join(current_dir, 'Week4')
sys.path.append(week4_dir)

from inference_pipeline import NewsQASystem

def run_demo_questions(qa_system, verbose=False):
    """Run a set of demo questions to showcase the system."""
    
    demo_questions = [
        "What is artificial intelligence?",
        "What is quantum computing?", 
        "Who is mentioned in the OpenAI article?",
        "What are the security risks mentioned?",
        "When was the latest AI model released?",
        "What is the main topic about open source?",
        "How does machine learning work?",
        "What companies are developing AI?",
        "What are the recent developments in technology?",
        "Who invented the computer?"  # This should trigger fallback
    ]
    
    print("LSTM News QA System - Demo")
    print("=" * 50)
    print(f"Running {len(demo_questions)} demo questions...")
    print("=" * 50)
    
    results = []
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 30)
        
        try:
            answer = qa_system.answer_question(question, verbose=verbose)
            display_text = qa_system.citation_formatter.format_display_text(answer)
            print(display_text)
            
            results.append({
                'question': question,
                'answer': answer,
                'success': True
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'question': question,
                'error': str(e),
                'success': False
            })
        
        print("-" * 50)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nDemo Summary:")
    print(f"Total questions: {len(demo_questions)}")
    print(f"Successful answers: {successful}")
    print(f"Failed answers: {len(demo_questions) - successful}")
    
    # Count fallback responses
    fallbacks = sum(1 for r in results if r['success'] and r['answer'].get('is_fallback', False))
    confident_answers = successful - fallbacks
    
    print(f"Confident answers: {confident_answers}")
    print(f"Fallback responses: {fallbacks}")
    
    return results

def interactive_mode(qa_system):
    """Run interactive question-answering mode."""
    
    print("LSTM News QA System - Interactive Mode")
    print("=" * 45)
    print("Ask questions about the news articles.")
    print("Type 'help' for commands, 'quit' to exit.")
    print("=" * 45)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\nAvailable commands:")
                print("  help    - Show this help message")
                print("  quit    - Exit the program")
                print("  verbose - Toggle verbose output")
                print("\nExample questions:")
                print("  - What is artificial intelligence?")
                print("  - Who developed the latest AI model?")
                print("  - What are the security concerns mentioned?")
                continue
            
            if question.lower() == 'verbose':
                # Toggle verbose mode (simplified)
                print("Verbose mode toggled (restart to apply)")
                continue
            
            print("\nProcessing your question...")
            
            start_time = datetime.now()
            answer = qa_system.answer_question(question, verbose=False)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            display_text = qa_system.citation_formatter.format_display_text(answer)
            print(display_text)
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError processing question: {e}")
            print("Please try again with a different question.")

def check_system_status(model_dir):
    """Check if all required models and data are available."""
    
    print("Checking system status...")
    print("-" * 30)
    
    required_files = [
        ('Processed passages', os.path.join(model_dir, '..', 'data', 'processed_passages.json')),
        ('Embedding manager', os.path.join(model_dir, 'embeddings.pkl')),
        ('Retriever model', os.path.join(model_dir, 'best_model.pth')),
        ('FAISS index', os.path.join(model_dir, 'faiss_index.index')),
        ('Reader tokenizer', os.path.join(model_dir, 'reader_tokenizer.pkl')),
        ('Reader model', os.path.join(model_dir, 'best_reader_news.pth')),
    ]
    
    status = {}
    all_good = True
    
    for name, filepath in required_files:
        exists = os.path.exists(filepath)
        status[name] = exists
        
        if exists:
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Missing ({filepath})")
            all_good = False
    
    print("-" * 30)
    
    if all_good:
        print("✅ All required files found! System ready.")
    else:
        print("❌ Some required files are missing.")
        print("\nTo fix this, please run:")
        print("1. Week 1: python Week1/data_processing.py")
        print("2. Week 2: python Week2/train_retriever.py")
        print("3. Week 3: python Week3/train_reader.py")
    
    return all_good

def main():
    """Main demo function."""
    
    parser = argparse.ArgumentParser(description='LSTM News QA System Demo')
    parser.add_argument('question', nargs='?', help='Single question to answer')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Run demo questions')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--confidence_threshold', type=float, default=0.3,
                       help='Confidence threshold for answers')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check system status')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, args.model_dir)
    
    print("LSTM-based News Question Answering System")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Model directory: {model_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print()
    
    # Check system status if requested
    if args.check:
        system_ready = check_system_status(model_dir)
        if not system_ready:
            return 1
        print()
    
    try:
        # Initialize QA system
        print("Initializing QA system...")
        qa_system = NewsQASystem(
            model_dir=model_dir,
            confidence_threshold=args.confidence_threshold
        )
        print("QA system initialized successfully!\n")
        
        # Run based on arguments
        if args.question:
            # Single question mode
            print(f"Question: {args.question}")
            print("Processing...")
            
            answer = qa_system.answer_question(args.question, verbose=args.verbose)
            display_text = qa_system.citation_formatter.format_display_text(answer)
            
            print("\n" + "=" * 60)
            print(display_text)
            print("=" * 60)
            
        elif args.interactive:
            # Interactive mode
            interactive_mode(qa_system)
            
        elif args.demo:
            # Demo mode
            run_demo_questions(qa_system, verbose=args.verbose)
            
        else:
            # Default: run a few sample questions
            print("Running sample questions (use --help for more options)...")
            
            sample_questions = [
                "What is artificial intelligence?",
                "What is quantum computing?",
                "Who developed OpenAI?"
            ]
            
            for i, question in enumerate(sample_questions, 1):
                print(f"\n{i}. Question: {question}")
                print("-" * 40)
                
                answer = qa_system.answer_question(question, verbose=args.verbose)
                display_text = qa_system.citation_formatter.format_display_text(answer)
                print(display_text)
                print("-" * 40)
            
            print(f"\nFor more options, run: python demo.py --help")
            print(f"Interactive mode: python demo.py --interactive")
            print(f"Full demo: python demo.py --demo")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all training steps (Week 1-3) have been completed")
        print("2. Check that model files exist in the models directory")
        print("3. Verify that cleaned_data.json is in the project root")
        print("4. Run with --check to see detailed status")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())