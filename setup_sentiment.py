#!/usr/bin/env python3
"""
Unified Setup Script for GRPO Agent Framework with Sentiment Analysis

This script provides a complete installation solution that combines:
1. Package installation with configurable dependency levels
2. Automated corpora downloads
3. Installation validation and testing
4. Comprehensive error handling and user guidance

Usage:
    python setup_sentiment.py [--level basic|advanced|full]
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SentimentSetup:
    """Unified sentiment analysis setup manager"""
    
    # Define dependency levels
    DEPENDENCY_LEVELS = {
        "basic": {
            "description": "Core sentiment analysis (TextBlob, VADER, NLTK)",
            "pip_extras": "",
            "dependencies": [
                "textblob>=0.17.1",
                "vaderSentiment>=3.3.2", 
                "nltk>=3.8.1"
            ]
        },
        "advanced": {
            "description": "Advanced NLP capabilities (+ spaCy, pandas, scikit-learn, emoji, flair)",
            "pip_extras": "[sentiment]",
            "dependencies": [
                "spacy>=3.4.0",
                "pandas>=1.5.0",
                "scikit-learn>=1.1.0",
                "emoji>=2.2.0",
                "flair>=0.12.2"
            ]
        },
        "full": {
            "description": "Complete NLP suite (+ sentence-transformers, advanced transformers)",
            "pip_extras": "[sentiment-full]",
            "dependencies": [
                "sentence-transformers>=2.2.0",
                "transformers[sentencepiece]>=4.21.0"
            ]
        }
    }
    
    def __init__(self, level: str = "advanced", verbose: bool = True):
        self.level = level
        self.verbose = verbose
        self.success_count = 0
        self.total_steps = 0
        
    def run_command(self, cmd: List[str], description: str, critical: bool = True) -> bool:
        """Run a command and handle errors"""
        if self.verbose:
            print(f"Running: {description}")
            print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self.verbose:
                print(f"✓ {description} completed successfully")
                if result.stdout and len(result.stdout.strip()) > 0:
                    # Only show first few lines of output to avoid spam
                    output_lines = result.stdout.strip().split('\n')
                    if len(output_lines) <= 3:
                        print(f"Output: {result.stdout.strip()}")
                    else:
                        print(f"Output: {output_lines[0]}... ({len(output_lines)} lines)")
            return True
            
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"✗ {description} failed")
                if e.stderr:
                    print(f"Error: {e.stderr}")
                if not critical:
                    print("  (Non-critical error - continuing)")
            
            return not critical  # Return True for non-critical errors


            return not critical  # Return True for non-critical errors
    
    def install_package_dependencies(self) -> bool:
        """Install the main package with specified sentiment analysis level"""
        level_config = self.DEPENDENCY_LEVELS[self.level]
        
        print(f"Installing GRPO Agent Framework with {self.level} sentiment analysis...")
        print(f"Description: {level_config['description']}")
        
        # Build pip install command
        pip_extras = level_config['pip_extras']
        install_target = f".{pip_extras}" if pip_extras else "."
        
        success = self.run_command(
            [sys.executable, "-m", "pip", "install", "-e", install_target],
            f"Installing package with {self.level} sentiment analysis dependencies"
        )
        
        return success
    
    def download_nltk_data(self) -> bool:
        """Download required NLTK data"""
        print("\nDownloading NLTK data...")
        
        # Essential NLTK downloads
        nltk_downloads = [
            ("punkt", "NLTK punkt tokenizer"),
            ("vader_lexicon", "NLTK VADER lexicon"),
            ("stopwords", "NLTK stopwords"),
            ("wordnet", "NLTK WordNet"),
            ("averaged_perceptron_tagger", "NLTK POS tagger"),
        ]
        
        success_count = 0
        for download_name, description in nltk_downloads:
            cmd = [sys.executable, "-c", f"import nltk; nltk.download('{download_name}')"]
            if self.run_command(cmd, description, critical=False):
                success_count += 1
        
        # Consider it successful if we got at least the essential ones
        essential_count = 2  # punkt and vader_lexicon are essential
        return success_count >= essential_count
    
    def download_textblob_corpora(self) -> bool:
        """Download TextBlob corpora"""
        print("\nDownloading TextBlob corpora...")
        
        success = self.run_command(
            [sys.executable, "-m", "textblob.download_corpora"],
            "TextBlob corpora download",
            critical=False
        )
        
        return success
    
    def test_sentiment_analysis(self) -> bool:
        """Test that sentiment analysis is working"""
        print("\nTesting sentiment analysis installation...")
        
        # Test core components
        tests = [
            self._test_textblob,
            self._test_vader,
            self._test_nltk,
        ]
        
        # Add advanced tests based on level
        if self.level in ["advanced", "full"]:
            tests.extend([
                self._test_pandas,
                self._test_sklearn,
            ])
        
        if self.level == "full":
            tests.extend([
                self._test_sentence_transformers,
            ])
        
        success_count = 0
        for test_func in tests:
            if test_func():
                success_count += 1
        
        # Require at least core components to work
        min_required = 3  # TextBlob, VADER, NLTK
        return success_count >= min_required
    
    def _test_textblob(self) -> bool:
        """Test TextBlob functionality"""
        test_script = '''
try:
    from textblob import TextBlob
    blob = TextBlob("I love this framework!")
    sentiment = blob.sentiment
    print(f"TextBlob sentiment: polarity={sentiment.polarity:.3f}, subjectivity={sentiment.subjectivity:.3f}")
    print("✓ TextBlob working")
except Exception as e:
    print(f"✗ TextBlob error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing TextBlob",
            critical=False
        )
    
    def _test_vader(self) -> bool:
        """Test VADER functionality"""
        test_script = '''
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores("I love this framework!")
    print(f"VADER sentiment: {scores}")
    print("✓ VADER working")
except Exception as e:
    print(f"✗ VADER error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing VADER",
            critical=False
        )
    
    def _test_nltk(self) -> bool:
        """Test NLTK functionality"""
        test_script = '''
try:
    import nltk
    tokens = nltk.word_tokenize("This is a test.")
    print(f"NLTK tokenization: {tokens}")
    print("✓ NLTK working")
except Exception as e:
    print(f"✗ NLTK error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing NLTK",
            critical=False
        )
    
    def _test_pandas(self) -> bool:
        """Test pandas functionality"""
        test_script = '''
try:
    import pandas as pd
    df = pd.DataFrame({"text": ["good", "bad"], "sentiment": [1, 0]})
    print(f"Pandas working: {len(df)} rows")
    print("✓ Pandas working")
except Exception as e:
    print(f"✗ Pandas error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing pandas",
            critical=False
        )
    
    def _test_sklearn(self) -> bool:
        """Test scikit-learn functionality"""
        test_script = '''
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(["good text", "bad text"])
    print(f"Scikit-learn working: {vectors.shape}")
    print("✓ Scikit-learn working")
except Exception as e:
    print(f"✗ Scikit-learn error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing scikit-learn",
            critical=False
        )
    
    def _test_sentence_transformers(self) -> bool:
        """Test sentence-transformers functionality"""
        test_script = '''
try:
    from sentence_transformers import SentenceTransformer
    # Use a small model for testing
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(["test sentence"])
    print(f"Sentence-transformers working: {embeddings.shape}")
    print("✓ Sentence-transformers working")
except Exception as e:
    print(f"✗ Sentence-transformers error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing sentence-transformers",
            critical=False
        )
    
    def test_reward_system_integration(self) -> bool:
        """Test that the reward system works with sentiment analysis"""
        print("\nTesting reward system integration...")
        
        test_script = '''
try:
    from rewards.multi_objective_reward import create_customer_service_reward
    
    # Test creating a sentiment-aware reward function
    reward_func = create_customer_service_reward(use_sentiment_analysis=True)
    print("✓ Reward system integration successful")
    
    # Test basic functionality
    from core.trajectory import ConversationTurn
    turns = [
        ConversationTurn(role="user", content="I'm frustrated with this service!"),
        ConversationTurn(role="assistant", content="I understand your frustration and I'm here to help resolve this issue.")
    ]
    
    import asyncio
    result = asyncio.run(reward_func.compute_reward(turns))
    print(f"✓ Sentiment-aware reward computation successful: score={result.score:.3f}")
    
except Exception as e:
    print(f"✗ Reward system integration error: {e}")
    raise
'''
        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing reward system integration",
            critical=False
        )


        return self.run_command(
            [sys.executable, "-c", test_script],
            "Testing reward system integration",
            critical=False
        )
    
    def setup_sentiment_analysis(self) -> bool:
        """Main setup function"""
        print("🚀 Setting up GRPO Agent Framework with Sentiment Analysis")
        print(f"📊 Level: {self.level.upper()} - {self.DEPENDENCY_LEVELS[self.level]['description']}")
        print("=" * 70)
        
        # Check if we're in the right directory
        if not os.path.exists("setup.py"):
            print("❌ setup.py not found. Please run this script from the project root directory.")
            return False
        
        # Define setup steps
        steps = [
            ("Installing package dependencies", self.install_package_dependencies),
            ("Downloading NLTK data", self.download_nltk_data),
            ("Downloading TextBlob corpora", self.download_textblob_corpora),
            ("Testing sentiment analysis components", self.test_sentiment_analysis),
            ("Testing reward system integration", self.test_reward_system_integration),
        ]
        
        self.total_steps = len(steps)
        self.success_count = 0
        
        for step_name, step_func in steps:
            print(f"\n📦 Step {self.success_count + 1}/{self.total_steps}: {step_name}")
            print("-" * 50)
            
            if step_func():
                self.success_count += 1
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
                if step_name == "Installing package dependencies":
                    print("   Cannot continue without package installation.")
                    break
                else:
                    print("   Continuing with remaining steps...")
        
        # Final report
        print(f"\n{'=' * 70}")
        print(f"Setup completed: {self.success_count}/{self.total_steps} steps successful")
        
        if self.success_count >= 4:  # At least core functionality working
            print("🎉 Sentiment analysis setup completed successfully!")
            print(f"\nInstalled level: {self.level.upper()}")
            self._print_usage_examples()
            self._print_available_dependencies()
            return True
        else:
            print("⚠️  Some steps failed. See troubleshooting guide below.")
            self._print_troubleshooting_guide()
            return False
    
    def _print_usage_examples(self):
        """Print usage examples"""
        print("\nUsage Examples:")
        print("=" * 15)
        print("from rewards.multi_objective_reward import create_customer_service_reward")
        print("from core.trajectory import ConversationTurn")
        print("")
        print("# Create sentiment-aware reward function")
        print("reward_func = create_customer_service_reward(use_sentiment_analysis=True)")
        print("")
        print("# Test with conversation")
        print("turns = [")
        print('    ConversationTurn(role="user", content="I\'m frustrated!"),')
        print('    ConversationTurn(role="assistant", content="I understand and will help.")')
        print("]")
        print("result = await reward_func.compute_reward(turns)")
        print("print(f'Score: {result.score:.3f}')")
    
    def _print_available_dependencies(self):
        """Print information about available dependency levels"""
        print(f"\nInstalled Components ({self.level}):")
        print("=" * 25)
        
        level_config = self.DEPENDENCY_LEVELS[self.level]
        base_deps = self.DEPENDENCY_LEVELS["basic"]["dependencies"]
        
        print("Core dependencies:")
        for dep in base_deps:
            print(f"  ✓ {dep}")
        
        if self.level in ["advanced", "full"]:
            advanced_deps = self.DEPENDENCY_LEVELS["advanced"]["dependencies"]
            print("\nAdvanced dependencies:")
            for dep in advanced_deps:
                print(f"  ✓ {dep}")
        
        if self.level == "full":
            full_deps = self.DEPENDENCY_LEVELS["full"]["dependencies"]
            print("\nFull suite dependencies:")
            for dep in full_deps:
                print(f"  ✓ {dep}")
        
        print(f"\nTo upgrade to a higher level:")
        if self.level == "basic":
            print("  python setup_sentiment.py --level advanced")
            print("  python setup_sentiment.py --level full")
        elif self.level == "advanced":
            print("  python setup_sentiment.py --level full")
    
    def _print_troubleshooting_guide(self):
        """Print troubleshooting information"""
        print("\nTroubleshooting Guide:")
        print("=" * 22)
        print("1. Manual installation:")
        print("   pip install textblob vaderSentiment nltk")
        print("   python -m textblob.download_corpora")
        print('   python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'vader_lexicon\')"')
        print("")
        print("2. If NLTK downloads fail:")
        print('   python -c "import nltk; nltk.download(\'all\')"')
        print("")
        print("3. If advanced dependencies fail:")
        print("   pip install spacy pandas scikit-learn")
        print("")
        print("4. Restart setup:")
        print(f"   python setup_sentiment.py --level {self.level}")


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Setup GRPO Agent Framework with Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dependency Levels:
  basic    - Core sentiment analysis (TextBlob, VADER, NLTK)
  advanced - + Advanced NLP (spaCy, pandas, scikit-learn, emoji, flair)  
  full     - + Complete suite (sentence-transformers, advanced transformers)

Examples:
  python setup_sentiment.py                    # Default: advanced level
  python setup_sentiment.py --level basic     # Minimal installation
  python setup_sentiment.py --level full      # Complete installation
  python setup_sentiment.py --quiet           # Minimal output
        """
    )
    
    parser.add_argument(
        "--level",
        choices=["basic", "advanced", "full"],
        default="advanced",
        help="Dependency level to install (default: advanced)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Create setup manager
    setup = SentimentSetup(level=args.level, verbose=not args.quiet)
    
    # Run setup
    success = setup.setup_sentiment_analysis()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
