# Unified Sentiment Analysis Setup

The GRPO Agent Framework now includes a unified setup system that combines package installation, dependency management, and validation into a single comprehensive script.

## Quick Start

```bash
# Default installation (recommended for most users)
python setup_sentiment.py

# Or specify the level explicitly
python setup_sentiment.py --level advanced
```

## Installation Levels

The unified setup supports three dependency levels:

### 1. Basic Level
**Command:** `python setup_sentiment.py --level basic`

**Includes:**
- `textblob>=0.17.1` - Basic sentiment analysis
- `vaderSentiment>=3.3.2` - Social media optimized sentiment
- `nltk>=3.8.1` - Natural language processing toolkit

**Best For:** Minimal deployments, quick testing, resource-constrained environments

### 2. Advanced Level (Default)
**Command:** `python setup_sentiment.py --level advanced`

**Includes:** Basic level PLUS:
- `spacy>=3.4.0` - Industrial-strength NLP
- `pandas>=1.5.0` - Data manipulation and analysis
- `scikit-learn>=1.1.0` - Machine learning utilities
- `emoji>=2.2.0` - Emoji sentiment analysis
- `flair>=0.12.2` - Named entity recognition and sentiment

**Best For:** Production deployments, comprehensive sentiment analysis, most use cases

### 3. Full Level
**Command:** `python setup_sentiment.py --level full`

**Includes:** Advanced level PLUS:
- `sentence-transformers>=2.2.0` - Semantic similarity and embeddings
- `transformers[sentencepiece]>=4.21.0` - Advanced transformer models

**Best For:** Research, cutting-edge features, maximum capability

## Features

### 🚀 **Automated Installation**
- Installs correct dependencies for chosen level
- Downloads required corpora (NLTK, TextBlob)
- Handles errors gracefully with detailed feedback

### 🧪 **Comprehensive Testing**
- Validates each component works correctly
- Tests integration with reward system
- Provides specific error diagnostics

### 📊 **Smart Error Handling**
- Non-critical errors don't stop the process
- Clear troubleshooting guidance
- Manual installation fallbacks

### 🔧 **Flexible Configuration**
- Multiple dependency levels
- Quiet mode for automated deployments
- Upgrade path between levels

## Usage Examples

### Basic Usage
```bash
# Install with default settings
python setup_sentiment.py

# Quiet installation for scripts
python setup_sentiment.py --quiet

# Minimal installation
python setup_sentiment.py --level basic
```

### After Installation
```python
from rewards.multi_objective_reward import create_customer_service_reward
from core.trajectory import ConversationTurn

# Create sentiment-aware reward function
reward_func = create_customer_service_reward(use_sentiment_analysis=True)

# Test with conversation
turns = [
    ConversationTurn(role="user", content="I'm really frustrated with this service!"),
    ConversationTurn(role="assistant", content="I understand your frustration and I'm here to help resolve this issue.")
]

# Compute sentiment-aware reward
result = await reward_func.compute_reward(turns)
print(f"Reward score: {result.score:.3f}")
```

## Upgrade Path

You can upgrade your installation level at any time:

```bash
# Start with basic
python setup_sentiment.py --level basic

# Upgrade to advanced
python setup_sentiment.py --level advanced

# Upgrade to full capabilities
python setup_sentiment.py --level full
```

## Troubleshooting

### Common Issues

**NLTK Download Errors:**
```bash
python -c "import nltk; nltk.download('all')"
```

**TextBlob Corpora Missing:**
```bash
python -m textblob.download_corpora
```

**Package Installation Fails:**
```bash
pip install --upgrade pip
python setup_sentiment.py --level basic
```

### Manual Installation

If the automated setup fails, you can install manually:

```bash
# Basic dependencies
pip install textblob vaderSentiment nltk

# Download corpora
python -m textblob.download_corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"

# Advanced dependencies (optional)
pip install spacy pandas scikit-learn emoji flair

# Full dependencies (optional)
pip install sentence-transformers "transformers[sentencepiece]"
```

## Validation

Test your installation:

```python
# Quick test
from rewards.multi_objective_reward import create_customer_service_reward
reward = create_customer_service_reward(use_sentiment_analysis=True)
print("✅ Sentiment analysis ready!")

# Detailed test
python setup_sentiment.py --level basic  # This includes validation
```

## Integration with Setup.py

The unified setup is fully integrated with the package's setup.py:

- **Basic dependencies** are in `install_requires` (always installed)
- **Advanced dependencies** are in `extras_require["sentiment"]`
- **Full dependencies** are in `extras_require["sentiment-full"]`

You can also use standard pip commands:
```bash
pip install -e .                    # Basic
pip install -e .[sentiment]         # Advanced  
pip install -e .[sentiment-full]    # Full
```

But the unified setup script (`setup_sentiment.py`) is recommended as it handles corpora downloads and validation automatically.
