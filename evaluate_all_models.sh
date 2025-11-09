#!/bin/bash

# Script to evaluate all models in models/comparison_20251107_094601

MODELS_DIR="models/comparison_20251107_094601"
EVALUATION_SCRIPT="src/evaluate_model.py"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting batch evaluation of models in ${MODELS_DIR}${NC}\n"

# Find all .pt files and evaluate them
for model_path in $(find "${MODELS_DIR}" -name "*.pt" -type f | sort); do
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}Evaluating: ${model_path}${NC}"
    echo -e "${GREEN}================================================${NC}"
    
    python "${EVALUATION_SCRIPT}" --model_path "${model_path}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed successfully for ${model_path}${NC}\n"
    else
        echo -e "${RED}✗ Evaluation failed for ${model_path}${NC}\n"
    fi
done

echo -e "${BLUE}All evaluations completed!${NC}"
echo -e "${BLUE}Results saved to data/03_results/${NC}"
