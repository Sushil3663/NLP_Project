"""
Complete pipeline runner for the LSTM News QA System.
Executes all training steps from Week 1 to Week 4 in sequence.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import argparse

def run_command(command, cwd=None, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {command}")
    print(f"Directory: {cwd or 'current'}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nCommand completed in {duration:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout[-1000:])  # Last 1000 chars
        else:
            print("❌ FAILED")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-1000:])  # Last 1000 chars
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT - Command took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return False

def check_prerequisites():
    """Check if prerequisites are available."""
    print("Checking prerequisites...")
    
    # Check Python packages
    required_packages = [
        'torch', 'numpy', 'pandas', 'nltk', 'spacy', 
        'scikit-learn', 'tqdm', 'requests', 'faiss-cpu'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check for cleaned_data.json
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(project_root, 'cleaned_data.json')
    
    if os.path.exists(data_file):
        print(f"✅ cleaned_data.json found")
    else:
        print(f"❌ cleaned_data.json not found at {data_file}")
        return False
    
    print("✅ All prerequisites satisfied")
    return True

def run_week1(project_root, skip_if_exists=True):
    """Run Week 1: Data Processing."""
    print(f"\n{'#'*60}")
    print("WEEK 1: DATA INGESTION & CLEANING")
    print(f"{'#'*60}")
    
    # Check if already completed
    output_file = os.path.join(project_root, 'data', 'processed_passages.json')
    if skip_if_exists and os.path.exists(output_file):
        print(f"✅ Week 1 already completed - {output_file} exists")
        return True
    
    week1_dir = os.path.join(project_root, 'Week1')
    command = f"{sys.executable} data_processing.py"
    
    return run_command(command, cwd=week1_dir, description="Week 1 - Data Processing")

def run_week2(project_root, skip_if_exists=True):
    """Run Week 2: BiLSTM Retriever Training."""
    print(f"\n{'#'*60}")
    print("WEEK 2: BiLSTM RETRIEVER TRAINING")
    print(f"{'#'*60}")
    
    # Check if already completed
    model_file = os.path.join(project_root, 'models', 'best_model.pth')
    index_file = os.path.join(project_root, 'models', 'faiss_index.index')
    
    if skip_if_exists and os.path.exists(model_file) and os.path.exists(index_file):
        print(f"✅ Week 2 already completed - models exist")
        return True
    
    week2_dir = os.path.join(project_root, 'Week2')
    command = f"{sys.executable} train_retriever.py --epochs 10 --batch_size 16"
    
    return run_command(command, cwd=week2_dir, description="Week 2 - Retriever Training")

def run_week3(project_root, skip_if_exists=True):
    """Run Week 3: LSTM + Attention Reader Training."""
    print(f"\n{'#'*60}")
    print("WEEK 3: LSTM + ATTENTION READER TRAINING")
    print(f"{'#'*60}")
    
    # Check if already completed
    reader_files = [
        os.path.join(project_root, 'models', 'best_reader_news.pth'),
        os.path.join(project_root, 'models', 'best_reader_squad.pth'),
        os.path.join(project_root, 'models', 'reader_tokenizer.pkl')
    ]
    
    if skip_if_exists and any(os.path.exists(f) for f in reader_files):
        print(f"✅ Week 3 already completed - reader models exist")
        return True
    
    week3_dir = os.path.join(project_root, 'Week3')
    command = f"{sys.executable} train_reader.py --squad_epochs 3 --news_epochs 5 --batch_size 8"
    
    return run_command(command, cwd=week3_dir, description="Week 3 - Reader Training")

def run_week4_demo(project_root):
    """Run Week 4: Demo the complete system."""
    print(f"\n{'#'*60}")
    print("WEEK 4: INFERENCE PIPELINE DEMO")
    print(f"{'#'*60}")
    
    # Run demo
    command = f"{sys.executable} demo.py --demo"
    
    return run_command(command, cwd=project_root, description="Week 4 - System Demo")

def generate_final_report(project_root, results):
    """Generate final project report."""
    report_file = os.path.join(project_root, 'outputs', 'final_project_report.txt')
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("LSTM-based News Question Answering System - Final Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PROJECT OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write("This project implements a complete news QA system with:\n")
        f.write("1. Data ingestion and cleaning (Week 1)\n")
        f.write("2. BiLSTM retriever with FAISS indexing (Week 2)\n")
        f.write("3. LSTM + attention reader (Week 3)\n")
        f.write("4. End-to-end inference pipeline (Week 4)\n\n")
        
        f.write("EXECUTION RESULTS\n")
        f.write("-" * 20 + "\n")
        for week, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            f.write(f"{week}: {status}\n")
        
        f.write("\nSYSTEM ARCHITECTURE\n")
        f.write("-" * 20 + "\n")
        f.write("Input Question → Preprocessing → Retrieval → Re-ranking → Reading → Answer + Citation\n\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 20 + "\n")
        
        # List key output files
        key_files = [
            'data/processed_passages.json',
            'models/embeddings.pkl',
            'models/best_model.pth',
            'models/faiss_index.index',
            'models/reader_tokenizer.pkl',
            'models/best_reader_news.pth',
            'outputs/week1_report.txt',
            'outputs/week2_metrics.json',
            'outputs/week3_evaluation_report.txt'
        ]
        
        for file_path in key_files:
            full_path = os.path.join(project_root, file_path)
            exists = "✅" if os.path.exists(full_path) else "❌"
            f.write(f"{exists} {file_path}\n")
        
        f.write(f"\nUSAGE INSTRUCTIONS\n")
        f.write("-" * 20 + "\n")
        f.write("1. Interactive mode: python demo.py --interactive\n")
        f.write("2. Single question: python demo.py 'What is AI?'\n")
        f.write("3. Demo questions: python demo.py --demo\n")
        f.write("4. Check status: python demo.py --check\n\n")
        
        f.write("PERFORMANCE NOTES\n")
        f.write("-" * 20 + "\n")
        f.write("- System uses GPU if available for faster processing\n")
        f.write("- Confidence threshold can be adjusted (default: 0.3)\n")
        f.write("- Fallback responses provided for low-confidence answers\n")
        f.write("- Citations include URL, headline, date, and publisher\n\n")
    
    print(f"📄 Final report generated: {report_file}")

def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description='LSTM News QA System - Complete Pipeline')
    parser.add_argument('--skip-week1', action='store_true', help='Skip Week 1 if already completed')
    parser.add_argument('--skip-week2', action='store_true', help='Skip Week 2 if already completed')
    parser.add_argument('--skip-week3', action='store_true', help='Skip Week 3 if already completed')
    parser.add_argument('--skip-demo', action='store_true', help='Skip final demo')
    parser.add_argument('--week-only', type=int, choices=[1,2,3,4], help='Run only specific week')
    parser.add_argument('--no-skip', action='store_true', help='Force re-run all steps')
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print("LSTM News QA System - Complete Pipeline Runner")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not satisfied. Please install required packages.")
        return 1
    
    skip_if_exists = not args.no_skip
    results = {}
    
    try:
        # Week 1: Data Processing
        if not args.week_only or args.week_only == 1:
            if not args.skip_week1:
                results['Week 1'] = run_week1(project_root, skip_if_exists)
            else:
                print("⏭️  Skipping Week 1")
                results['Week 1'] = True
        
        # Week 2: Retriever Training
        if not args.week_only or args.week_only == 2:
            if not args.skip_week2 and results.get('Week 1', True):
                results['Week 2'] = run_week2(project_root, skip_if_exists)
            else:
                if args.skip_week2:
                    print("⏭️  Skipping Week 2")
                    results['Week 2'] = True
                else:
                    print("❌ Skipping Week 2 due to Week 1 failure")
                    results['Week 2'] = False
        
        # Week 3: Reader Training
        if not args.week_only or args.week_only == 3:
            if not args.skip_week3 and results.get('Week 2', True):
                results['Week 3'] = run_week3(project_root, skip_if_exists)
            else:
                if args.skip_week3:
                    print("⏭️  Skipping Week 3")
                    results['Week 3'] = True
                else:
                    print("❌ Skipping Week 3 due to previous failures")
                    results['Week 3'] = False
        
        # Week 4: Demo
        if not args.week_only or args.week_only == 4:
            if not args.skip_demo and results.get('Week 3', True):
                results['Week 4'] = run_week4_demo(project_root)
            else:
                if args.skip_demo:
                    print("⏭️  Skipping Week 4 demo")
                    results['Week 4'] = True
                else:
                    print("❌ Skipping Week 4 due to previous failures")
                    results['Week 4'] = False
        
        # Generate final report
        generate_final_report(project_root, results)
        
        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        total_steps = len(results)
        successful_steps = sum(1 for success in results.values() if success)
        
        for week, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{week}: {status}")
        
        print(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully")
        
        if successful_steps == total_steps:
            print("🎉 All steps completed successfully!")
            print("\nNext steps:")
            print("1. Run: python demo.py --interactive")
            print("2. Or: python demo.py --demo")
            print("3. Or: python demo.py 'Your question here'")
        else:
            print("⚠️  Some steps failed. Check the output above for details.")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())