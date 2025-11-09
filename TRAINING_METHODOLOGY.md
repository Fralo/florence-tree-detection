# DeepForest Training Methodology

## Overview

This document explains the training approach used in this project for tree detection in aerial imagery. The training methodology employed is **Transfer Learning through Fine-Tuning**, specifically using a pre-trained object detection model adapted to your custom dataset.

## What Type of Training is This?

### Transfer Learning via Fine-Tuning

Your training script implements **fine-tuning**, which is a form of transfer learning. Here's what makes it fine-tuning:

1. **Starting Point**: Pre-trained weights from a previously trained model
2. **Adaptation**: Continuing training on your custom dataset
3. **Goal**: Adapt the model to detect trees in your specific aerial imagery

## Technical Details

### Base Model Architecture

The project uses **DeepForest**, which is built on top of **Faster R-CNN** (Region-based Convolutional Neural Network):

- **Backbone**: ResNet or similar CNN architecture for feature extraction
- **Detection Head**: Region Proposal Network (RPN) + ROI pooling + classification/regression heads
- **Framework**: PyTorch with PyTorch Lightning for training orchestration

### Pre-trained Model Source

According to the configuration and code:

```python
# Line 410-413 in train_model.py
model.model = torch.load(
    model_config["final_model_path"],
    weights_only=False
)
```

The model loads pre-trained weights from `models/deepforest_finetuned_3.pt`, which suggests you're starting from:
- Either the original DeepForest pre-trained weights (trained on aerial tree imagery)
- Or weights from a previous fine-tuning iteration

The commented-out code shows the original intended source:
```python
# model.load_model(model_name="weecology/deepforest-tree", revision="main")
```

This indicates the base model is **weecology/deepforest-tree**, a model pre-trained on large-scale aerial tree detection datasets.

## Fine-Tuning Process Breakdown

### 1. Model Initialization
- Load pre-trained model weights
- The model already knows how to detect trees from its pre-training
- All layers are initialized with learned parameters (not random)

### 2. Data Adaptation
Your custom dataset structure:
```
data/02_processed/
├── train/
│   ├── images/
│   └── annotations.csv
├── evaluate/
│   ├── images/
│   └── annotations.csv
└── test/
    ├── images/
    └── annotations.csv
```

### 3. Training Configuration

**Key Hyperparameters** (from `train_model.py:424-441`):

```python
batch_size: 4 (default)
epochs: 20 (default)
learning_rate: 0.0001
optimizer: (inherited from DeepForest, likely Adam or SGD)
scheduler: ReduceLROnPlateau
  - monitor: map (Mean Average Precision)
  - patience: 3
  - factor: 0.1
  - min_lr: 1e-6
```

**Why Low Learning Rate?**
- Fine-tuning uses a much lower learning rate (0.0001) compared to training from scratch
- This preserves the useful features learned during pre-training
- Only makes small adjustments to adapt to your specific dataset

### 4. Training Strategy

**Full Model Fine-Tuning**: All layers are trainable
- Unlike feature extraction (where you freeze early layers), this approach updates all parameters
- Allows both low-level features and high-level detection patterns to adapt

**Learning Rate Schedule**:
- ReduceLROnPlateau monitors validation mAP
- Reduces LR by 10x when performance plateaus
- Prevents overshooting optimal weights while allowing convergence

### 5. Data Augmentation Strategy

A unique feature of your implementation is the **configurable augmentation registry** (lines 71-101):

Available strategies:
1. **none**: No augmentation (baseline)
2. **light**: Basic flips and minor color adjustments
3. **medium**: Balanced approach (default, recommended)
4. **heavy**: Aggressive augmentation for small datasets
5. **aerial**: Optimized for aerial imagery with full rotations
6. **custom**: Experimental playground

**Default Medium Strategy** includes:
- Horizontal/vertical flips (p=0.5)
- Random 90° rotations (p=0.5)
- Shift/scale/rotate (±5% shift, ±10% scale, ±15° rotation)
- Brightness/contrast adjustments (±20%)
- Hue/saturation/value variations (simulating different weather/lighting)

**Why This Matters**:
- Augmentation increases effective dataset size
- Helps model generalize to various lighting conditions, orientations
- Prevents overfitting on limited training data

## Not Training From Scratch

This is definitively **NOT** training from scratch because:

1. ✅ Loads pre-trained weights: `torch.load(model_config["final_model_path"])`
2. ✅ Uses low learning rate (0.0001) typical of fine-tuning
3. ✅ Relatively few epochs (20) - training from scratch often needs 100+
4. ✅ No architecture modifications - uses existing DeepForest model
5. ✅ Builds on knowledge from pre-trained model

## Not Domain Adaptation

This is also **NOT** pure domain adaptation because:
- Domain adaptation typically keeps the source domain data
- You're fully fine-tuning on your target domain only
- No adversarial training or domain-specific loss functions

## Fine-Tuning vs. Transfer Learning Terminology

**Transfer Learning** (broad category):
- Any method that transfers knowledge from one task to another
- Includes: feature extraction, fine-tuning, domain adaptation, etc.

**Fine-Tuning** (specific technique):
- A type of transfer learning
- Continues training a pre-trained model on new data
- Updates model weights through backpropagation

**Your Approach**: Fine-tuning with data augmentation

## Why Fine-Tuning Works for This Task

1. **Similar Task**: Pre-trained model already does tree detection in aerial imagery
2. **Limited Data**: Fine-tuning requires less data than training from scratch
3. **Feature Reuse**: Low-level features (edges, textures) transfer well
4. **Efficient**: Faster convergence, lower computational cost

## Model Evaluation Strategy

From `train_model.py:514-564`:

```python
def evaluate_model(model, config):
    # Batch size set to 1 for evaluation to handle variable image sizes
    results = model.evaluate(
        csv_file=config['val_csv'],
        root_dir=config.get('val_root_dir'),
        iou_threshold=0.4
    )
```

**Metrics**:
- **mAP** (Mean Average Precision): Primary metric monitored during training
- **IoU Threshold**: 0.4 for matching predictions to ground truth
- Precision/Recall curves (inherited from DeepForest evaluation)

## Key Training Workflow

```
1. Load pre-trained DeepForest model
   ↓
2. Load custom training/validation data
   ↓
3. Apply augmentation strategy
   ↓
4. Fine-tune for 20 epochs
   ↓
5. Monitor validation mAP
   ↓
6. Reduce learning rate on plateau
   ↓
7. Save best checkpoint
   ↓
8. Evaluate on test set
```

## Advanced Features

### 1. Safe Augmentation Wrapper (lines 29-68)
Handles edge cases where augmentation removes all bounding boxes:
```python
class SafeAlbumentationsWrapper:
    # Ensures proper tensor shapes even when N=0 boxes
    # Critical for preventing crashes during training
```

### 2. Checkpoint Management
- Saves best model based on validation mAP
- Keeps PyTorch Lightning checkpoint for resume capability
- Saves final model state dict for inference

### 3. Configuration Persistence
Saves training configuration alongside model:
- Augmentation strategy used
- Hyperparameters
- Data paths
- Enables reproducibility

## Summary

**Training Type**: **Transfer Learning via Fine-Tuning**

**Key Characteristics**:
- Starts from pre-trained DeepForest weights (trained on aerial tree imagery)
- Adapts to your specific aerial imagery dataset
- Uses low learning rate (0.0001) to preserve pre-trained knowledge
- Employs configurable data augmentation to improve generalization
- Monitors validation mAP with learning rate scheduling
- Saves best model based on validation performance

**Why This Approach**:
- More efficient than training from scratch
- Requires less labeled data
- Leverages existing tree detection knowledge
- Faster convergence (20 epochs vs. 100+ for scratch training)
- Better generalization on limited datasets

This fine-tuning approach is the industry standard for adapting pre-trained object detection models to custom datasets, especially when working with domain-specific data like aerial imagery where pre-trained models in the same domain are available.
