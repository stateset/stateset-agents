#!/bin/bash

# Production TRL GRPO Training Script for GRPO Agent Framework
# This script sets up the environment and runs production training
# for the openai/gpt-oss-120b model

set -e  # Exit on any error

echo "🚀 Starting Production TRL GRPO Training for GRPO Agent Framework"
echo "=============================================================="

# Set production environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=${WANDB_PROJECT:-"grpo-agent-trl-training"}

# Model configuration
export MODEL_NAME=${MODEL_NAME:-"openai/gpt-oss-120b"}

# Production training configuration - Optimized for memory efficiency
export MAX_EXAMPLES=${MAX_EXAMPLES:-5000}
export BATCH_SIZE=${BATCH_SIZE:-1}  # Small batch to avoid OOM
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-1}
export LEARNING_RATE=${LEARNING_RATE:-5e-6}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export NUM_EPISODES=${NUM_EPISODES:-1000}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
export WARMUP_STEPS=${WARMUP_STEPS:-100}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# Generation parameters
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-256}
export TEMPERATURE=${TEMPERATURE:-0.7}
export TOP_P=${TOP_P:-0.9}

# LoRA configuration for efficient training
export USE_LORA=${USE_LORA:-true}
export LORA_R=${LORA_R:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}

# GRPO specific settings
export BETA=${BETA:-0.0}  # KL penalty (0.0 = no penalty)
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}
export NUM_ITERATIONS=${NUM_ITERATIONS:-1}

# Memory optimization
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
export USE_BF16=${USE_BF16:-true}
export USE_FP16=${USE_FP16:-false}

# Logging and checkpointing
export LOGGING_STEPS=${LOGGING_STEPS:-10}
export SAVE_STEPS=${SAVE_STEPS:-100}
export EVAL_STEPS=${EVAL_STEPS:-50}

# Output directory
export OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/trl_grpo_production"}
mkdir -p $OUTPUT_DIR

# Data configuration
export DATA_PATH=${DATA_PATH:-"training_data.jsonl"}

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export OMP_NUM_THREADS=4

# Evaluation settings
export RUN_EVAL=${RUN_EVAL:-true}

# Quick mode for testing (reduces parameters significantly)
export QUICK_MODE=${QUICK_MODE:-false}

# Log system info
echo "📊 System Information:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected')"
echo "CUDA Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for required Python packages
echo "📦 Checking dependencies..."
python -c "
import sys
try:
    import torch
    import transformers
    import peft
    import datasets
    import trl
    print('✅ All required packages are installed')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    print('Please install: pip install torch transformers peft datasets trl wandb')
    sys.exit(1)
"

# Check if running in quick mode
if [ "$QUICK_MODE" = "true" ]; then
    echo "⚡ Running in QUICK MODE with reduced parameters"
    export NUM_EPISODES=10
    export MAX_EXAMPLES=100
    export NUM_GENERATIONS=2
    export SAVE_STEPS=10
    export REPORT_TO=none
fi

# Create a timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="trl_grpo_${MODEL_NAME//\//_}_${TIMESTAMP}"
export RUN_NAME

# Create run-specific output directory
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir -p $RUN_OUTPUT_DIR
export OUTPUT_DIR=$RUN_OUTPUT_DIR

# Save configuration
echo "💾 Saving configuration..."
cat > "${OUTPUT_DIR}/training_config.env" << EOF
# Training Configuration
MODEL_NAME=$MODEL_NAME
MAX_EXAMPLES=$MAX_EXAMPLES
BATCH_SIZE=$BATCH_SIZE
LEARNING_RATE=$LEARNING_RATE
NUM_EPOCHS=$NUM_EPOCHS
NUM_EPISODES=$NUM_EPISODES
GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS
USE_LORA=$USE_LORA
LORA_R=$LORA_R
LORA_ALPHA=$LORA_ALPHA
BETA=$BETA
NUM_GENERATIONS=$NUM_GENERATIONS
TIMESTAMP=$TIMESTAMP
EOF

echo "Configuration saved to: ${OUTPUT_DIR}/training_config.env"

# Clear GPU cache before training
if [ "$(nvidia-smi 2>/dev/null)" ]; then
    echo "🧹 Clearing GPU cache..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
fi

# Set Python path to include the framework
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))"

# Run training with error handling
echo "🔥 Starting training..."
echo "Output directory: $OUTPUT_DIR"
echo "Logging to: ${OUTPUT_DIR}/training_${TIMESTAMP}.log"

# Change to examples directory to run the training script
cd "$(dirname $(dirname $(realpath $0)))/examples"

# Run the training script
python train_with_trl_grpo.py 2>&1 | tee "${OUTPUT_DIR}/training_${TIMESTAMP}.log"

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Model saved in: ${OUTPUT_DIR}/final_model"
    echo "📋 Training log saved in: ${OUTPUT_DIR}/training_${TIMESTAMP}.log"
    
    # Display final model info if available
    if [ -d "${OUTPUT_DIR}/final_model" ]; then
        echo "🏆 Final model information:"
        ls -lh "${OUTPUT_DIR}/final_model" | head -5
    fi
    
    # Show evaluation results if available
    if [ -f "${OUTPUT_DIR}/evaluation_results.json" ]; then
        echo "📊 Evaluation results available at: ${OUTPUT_DIR}/evaluation_results.json"
        # Display first few results
        python -c "
import json
with open('${OUTPUT_DIR}/evaluation_results.json', 'r') as f:
    results = json.load(f)
    print('\nSample evaluation results:')
    for r in results[:3]:
        print(f'Q: {r[\"query\"]}')
        print(f'A: {r[\"response\"][:100]}...')
        print()
"
    fi
else
    echo "❌ Training failed! Check the log file for details."
    echo "Log file: ${OUTPUT_DIR}/training_${TIMESTAMP}.log"
    exit 1
fi

echo "🎯 Production training pipeline completed!"
echo "To view training progress in Weights & Biases (if enabled):"
echo "  wandb login"
echo "  Visit: https://wandb.ai/${WANDB_ENTITY:-your-entity}/${WANDB_PROJECT}" 