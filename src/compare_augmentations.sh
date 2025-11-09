#!/bin/bash

################################################################################
# Augmentation Strategy Comparison Script
# 
# This script trains the DeepForest model with each augmentation strategy
# to systematically compare their performance.
#
# Usage: ./src/compare_augmentations.sh
################################################################################

echo ""
echo "========================================================================"
echo "  DeepForest Augmentation Strategy Comparison Experiment"
echo "========================================================================"
echo ""
echo "This will train 6 models with different augmentation strategies."
echo "Each model will be trained for 20 epochs with default parameters."
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Configuration
EPOCHS=20
BATCH_SIZE=4
NUM_WORKERS=0
BASE_OUTPUT_DIR="models/comparison"

# Create timestamp for this experiment run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"

echo ""
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo ""

# Array of augmentation strategies
strategies=("none" "light" "medium" "heavy" "aerial" "custom")

# Counter for progress tracking
total=${#strategies[@]}
current=0

# Function to train with a specific strategy
train_strategy() {
    local strategy=$1
    current=$((current + 1))
    
    echo ""
    echo "========================================================================"
    echo "  Training ${current}/${total}: ${strategy} strategy"
    echo "========================================================================"
    echo ""
    
    output_dir="${EXPERIMENT_DIR}/${strategy}"
    
    # Train the model
    python src/train_model.py \
        --augmentation-strategy "${strategy}" \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --output-dir "${output_dir}"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Completed: ${strategy} strategy"
        echo "  Results saved to: ${output_dir}"
    else
        echo ""
        echo "✗ Failed: ${strategy} strategy (exit code: ${exit_code})"
        echo "  Continuing with next strategy..."
    fi
    
    echo ""
}

# Start experiment
start_time=$(date +%s)

# Train with each strategy
for strategy in "${strategies[@]}"; do
    train_strategy "${strategy}"
done

# End experiment
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "========================================================================"
echo "  Experiment Completed!"
echo "========================================================================"
echo ""
echo "Total training time: ${hours}h ${minutes}m ${seconds}s"
echo "Results location: ${EXPERIMENT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Compare the config_*.txt files in each strategy directory"
echo "  2. Evaluate models on test set using src/evaluate_model.py"
echo "  3. Visualize predictions to compare quality"
echo ""
echo "To view results:"
echo "  ls -lh ${EXPERIMENT_DIR}/"
echo ""
echo "========================================================================"
