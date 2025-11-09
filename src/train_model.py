"""
DeepForest Fine-tuning Script
Fine-tunes the DeepForest model on custom tree detection data.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from config import load_config
import torch
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2

config = load_config()
train_config = config["training"]
model_config = config["model"]
data_config = config["data"]


# ============================================================================
# AUGMENTATION CONFIGURATIONS
# ============================================================================

class SafeAlbumentationsWrapper:
    """
    Wrapper for Albumentations transforms that handles edge cases where
    augmentations remove all bounding boxes (e.g., due to cropping or rotation).
    
    This ensures compatibility with DeepForest/PyTorch which expects tensors
    of shape [N, 4] even when N=0.
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image, bboxes, category_ids):
        """
        Apply augmentation and ensure proper tensor shapes even with empty boxes.
        
        Args:
            image: Input image (numpy array)
            bboxes: Bounding boxes in pascal_voc format
            category_ids: Category labels for each box
            
        Returns:
            dict with 'image', 'bboxes', and 'category_ids' keys
        """
        # Apply augmentation
        augmented = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        
        # Handle empty bounding boxes case
        # Albumentations may return empty list if all boxes are out of bounds
        # DeepForest expects numpy arrays with proper shape (N, 4) even when N=0
        if len(augmented['bboxes']) == 0:
            # Ensure proper 2D shape (0, 4) instead of (0,)
            import numpy as np
            augmented['bboxes'] = np.zeros((0, 4), dtype=np.float32)
            augmented['category_ids'] = np.array([], dtype=np.int64)
        
        return augmented


def get_augmentation_registry():
    """
    Registry of different augmentation strategies for experimentation.
    Each strategy is optimized for different scenarios.
    """
    return {
        'none': {
            'description': 'No augmentations - baseline',
            'transform': get_transform_none
        },
        'light': {
            'description': 'Light augmentations - basic flips and minor color adjustments',
            'transform': get_transform_light
        },
        'medium': {
            'description': 'Medium augmentations - balanced approach (RECOMMENDED)',
            'transform': get_transform_medium
        },
        'heavy': {
            'description': 'Heavy augmentations - aggressive for small datasets',
            'transform': get_transform_heavy
        },
        'aerial': {
            'description': 'Optimized for aerial imagery - rotations and lighting',
            'transform': get_transform_aerial
        },
        'custom': {
            'description': 'Custom augmentations - modify this for experiments',
            'transform': get_transform_custom
        }
    }


def get_transform_none(augment):
    """No augmentations - use as baseline for comparison."""
    transform = A.Compose([
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    return SafeAlbumentationsWrapper(transform)


def get_transform_light(augment):
    """
    Light augmentations - conservative approach.
    Good for: Large datasets, high-quality images, initial experiments
    """
    if augment:
        transform = A.Compose([
            # Basic geometric transformations
            A.HorizontalFlip(p=0.5),
            
            # Minimal color adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    
    return SafeAlbumentationsWrapper(transform)


def get_transform_medium(augment):
    """
    Medium augmentations - balanced approach (RECOMMENDED STARTING POINT).
    Good for: Most aerial tree detection scenarios
    """
    if augment:
        transform = A.Compose([
            # Geometric transformations (aerial imagery has no fixed orientation)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Small shifts and rotations
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5
            ),
            
            # Color augmentations for varying lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Simulate different weather/atmospheric conditions
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    
    return SafeAlbumentationsWrapper(transform)


def get_transform_heavy(augment):
    """
    Heavy augmentations - aggressive approach.
    Good for: Small datasets, highly variable conditions, preventing overfitting
    WARNING: May harm performance if dataset is already large and diverse
    """
    if augment:
        transform = A.Compose([
            # All geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=0,
                p=0.7
            ),
            
            # Aggressive color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            
            # Environmental effects
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    
    return SafeAlbumentationsWrapper(transform)


def get_transform_aerial(augment):
    """
    Optimized for aerial imagery - focuses on rotations and lighting.
    Good for: Drone/satellite imagery where orientation is arbitrary
    """
    if augment:
        transform = A.Compose([
            # All rotations are valid for aerial imagery
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=180, border_mode=0, p=0.4),  # Full rotation range
            
            # Scale variations (different flight altitudes)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.15,
                rotate_limit=0,  # Already handled by Rotate
                border_mode=0,
                p=0.4
            ),
            
            # Lighting variations (time of day, weather, seasons)
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.4
            ),
            
            # Note: RandomFog removed due to compatibility issues
            # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    
    return SafeAlbumentationsWrapper(transform)


def get_transform_custom(augment):
    """
    Custom augmentations - modify this for your specific experiments.
    This is your playground for testing new augmentation combinations.
    """
    if augment:
        transform = A.Compose([
            # ADD YOUR CUSTOM AUGMENTATIONS HERE
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Example: Try different combinations and parameters
            # A.Transpose(p=0.5),
            # A.ElasticTransform(p=0.2),
            # A.GridDistortion(p=0.2),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"], min_area=1.0, min_visibility=0.1))
    
    return SafeAlbumentationsWrapper(transform)


def get_augmentation_strategy(strategy_name):
    """
    Get augmentation transform function by strategy name.
    
    Args:
        strategy_name: Name of augmentation strategy from registry
    
    Returns:
        Transform function
    """
    registry = get_augmentation_registry()
    
    if strategy_name not in registry:
        available = ', '.join(registry.keys())
        raise ValueError(
            f"Unknown augmentation strategy: '{strategy_name}'\n"
            f"Available strategies: {available}"
        )
    
    return registry[strategy_name]['transform']


def print_augmentation_info():
    """Print information about available augmentation strategies."""
    print("\n" + "="*70)
    print(" " * 20 + "Available Augmentation Strategies")
    print("="*70)
    
    registry = get_augmentation_registry()
    for name, info in registry.items():
        print(f"\n  [{name}]")
        print(f"    {info['description']}")
    
    print("\n" + "="*70 + "\n")


def load_and_validate_data(train_csv, val_csv):
    """
    Load and validate training and validation data.
    
    Expected CSV format:
    image_path, xmin, ymin, xmax, ymax, label
    """
    print("\n" + "="*50)
    print("Loading and validating data...")
    print("="*50)
    
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Validate required columns
    required_columns = ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Training CSV missing required column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Validation CSV missing required column: {col}")
    
    print(f"✓ Training data: {len(train_df)} annotations, {train_df['image_path'].nunique()} images")
    print(f"✓ Validation data: {len(val_df)} annotations, {val_df['image_path'].nunique()} images")
    print(f"✓ Classes: {train_df['label'].unique()}")
    
    return train_df, val_df


def create_model(config):
    """
    Create and configure DeepForest model with augmentations.
    
    Args:
        config: Dictionary with model configuration including 'augmentation_strategy'
    """
    print("\n" + "="*50)
    print("Creating DeepForest model...")
    print("="*50)
    
    # Get augmentation strategy
    augmentation_strategy = config.get('augmentation_strategy', 'medium')
    transform_func = get_augmentation_strategy(augmentation_strategy)
    
    print(f"✓ Using augmentation strategy: '{augmentation_strategy}'")
    registry = get_augmentation_registry()
    print(f"  Description: {registry[augmentation_strategy]['description']}")
    
    # Create model with custom transforms
    model = main.deepforest(transforms=transform_func)
    
    # Load pretrained weights if specified
    # model.load_model(model_name="weecology/deepforest-tree", revision="main")
    
    # Load pretrained data with old weights 
    model.model = torch.load(
        model_config["final_model_path"],
        weights_only=False
    )
    
    # Configure model
    model.config["train"]["csv_file"] = config['train_csv']
    model.config["train"]["root_dir"] = config.get('train_root_dir', os.path.dirname(config['train_csv']))
    
    model.config["validation"]["csv_file"] = config['val_csv']
    model.config["validation"]["root_dir"] = config.get('val_root_dir', os.path.dirname(config['val_csv']))
    
    # Training hyperparameters
    model.config["batch_size"] = config.get('batch_size', 4)
    model.config["train"]["epochs"] = config.get('epochs', 20)
    model.config["train"]["lr"] = config.get('learning_rate', 0.0001)
    model.config["train"]["scheduler"] = {
        "type": "reduce_on_plateau",
        "monitor": "map",
        "params": {
            "patience": 3,
            "mode": "max",
            "factor": 0.1,
            "threshold": 0.0001,
            "threshold_mode": "rel",
            "cooldown": 1,
            "min_lr": 1e-6,
            "eps": 1e-8
        }
    }
    model.config["score_thresh"] = config.get('score_thresh', 0.4)
    model.config["nms_thresh"] = config.get('nms_thresh', 0.15)
    
    # Set number of workers for data loading
    model.config["workers"] = config.get('num_workers', 4)
    
    print(f"✓ Batch size: {model.config['batch_size']}")
    print(f"✓ Epochs: {model.config['train']['epochs']}")
    print(f"✓ Learning rate: {model.config['train']['lr']}")
    print(f"✓ Score threshold: {model.config['score_thresh']}")
    print(f"✓ NMS threshold: {model.config['nms_thresh']}")
    
    return model


def train_model(model, config):
    """
    Train the DeepForest model.
    
    Args:
        model: DeepForest model instance
        config: Dictionary with training configuration
    """
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Setup checkpoint callback to save best model
    output_dir = Path(config.get('output_dir', 'models'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename='best_model',
        monitor='map',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    
    # Setup trainer arguments
    trainer_args = {
        "fast_dev_run": config.get('fast_dev_run', False),
        "max_epochs": config.get('epochs', 20),
        "callbacks": [checkpoint_callback],
    }
    
    # Add GPU support if available
    if torch.cuda.is_available():
        print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = 1
    else:
        print("✓ Training on CPU/MPS")
        trainer_args["accelerator"] = "mps"
    
    # Train the model
    model.create_trainer(**trainer_args)
    model.trainer.fit(model)
    
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {checkpoint_callback.best_model_path}")
    
    # Load the best model weights
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"✓ Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    
    return best_model_path


def evaluate_model(model, config):
    """
    Evaluate the trained model on validation set.
    
    Args:
        model: Trained DeepForest model
        config: Dictionary with evaluation configuration
    """
    print("\n" + "="*50)
    print("Evaluating model...")
    print("="*50)
    
    # Save original batch size and set to 1 for evaluation to avoid collation issues
    # Images may have different sizes, and batch_size > 1 causes tensor stacking errors
    original_batch_size = model.config["batch_size"]
    model.config["batch_size"] = 1
    
    # Run evaluation
    results = model.evaluate(
        csv_file=config['val_csv'],
        root_dir=config.get('val_root_dir', os.path.dirname(config['val_csv'])),
        iou_threshold=config.get('iou_threshold', 0.4)
    )
    
    # Restore original batch size
    model.config["batch_size"] = original_batch_size
    
    print("\nEvaluation Results:")
    print("-" * 50)
    if results is not None:
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}:")
                print(value)
    
    # Save results
    results_file = Path(config.get('output_dir', 'results')) / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("DeepForest Evaluation Results\n")
        f.write("="*50 + "\n")
        if results is not None:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}:\n{value}\n\n")
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


def save_model(model, config):
    """
    Save the trained model with augmentation info.
    
    Args:
        model: Trained DeepForest model (already loaded with best weights)
        config: Dictionary with save configuration
    """
    print("\n" + "="*50)
    print("Saving model...")
    print("="*50)
    
    output_dir = Path(config.get('output_dir', 'models'))
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_models = len([f for f in os.listdir(output_dir) if f.endswith('.pt')])

    # Include augmentation strategy in filename
    aug_strategy = config.get('augmentation_strategy', 'default')
    model_path = output_dir / f"deepforest_finetuned_{existing_models}_{aug_strategy}.pt"
    
    # Save model state dict
    torch.save(model.model, model_path)
    
    print(f"✓ Model (best weights) saved to {model_path}")
    if config.get('best_model_path'):
        print(f"✓ PyTorch Lightning checkpoint also available at: {config['best_model_path']}")
    
    # Save configuration with augmentation info
    config_path = output_dir / f'config_{existing_models}_{aug_strategy}.txt'
    with open(config_path, 'w') as f:
        f.write("DeepForest Training Configuration\n")
        f.write("="*50 + "\n")
        f.write(f"Augmentation Strategy: {config.get('augmentation_strategy', 'none')}\n")
        registry = get_augmentation_registry()
        if config.get('augmentation_strategy') in registry:
            f.write(f"Strategy Description: {registry[config['augmentation_strategy']]['description']}\n")
        f.write("="*50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✓ Configuration saved to {config_path}")


def main_pipeline(args):
    """
    Main training pipeline.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print(" " * 15 + "DeepForest Fine-tuning Pipeline")
    print("="*70)
    
    # Print augmentation info if requested
    if args.list_augmentations:
        print_augmentation_info()
        return
    
    training_data = train_config["training_data"]
    training_annotations = train_config["training_annotations"]
    validation_data = train_config["validation_data"]
    validation_annotations = train_config["validation_annotations"]

    # Create configuration
    config = {
        'train_csv': training_annotations,
        'val_csv': validation_annotations,
        'train_root_dir': training_data,
        'val_root_dir': validation_data,
        'use_pretrained': args.use_pretrained,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'score_thresh': args.score_thresh,
        'nms_thresh': args.nms_thresh,
        'num_workers': args.num_workers,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'iou_threshold': args.iou_threshold,
        'fast_dev_run': False,
        'augmentation_strategy': args.augmentation_strategy,
    }
    
    # Load and validate data
    train_df, val_df = load_and_validate_data(training_annotations, validation_annotations)
    
    # Create model with augmentation strategy
    model = create_model(config)
    
    # Train model (returns path to best model checkpoint)
    best_model_path = train_model(model, config)
    config['best_model_path'] = best_model_path
    
    # Evaluate model (now using best weights)
    # evaluate_model(model, config)
    
    # Save final model with custom name
    save_model(model, config)
    
    print("\n" + "="*70)
    print(" " * 20 + "Pipeline completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepForest model for tree detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Augmentation arguments
    parser.add_argument(
        '--augmentation-strategy',
        type=str,
        default='medium',
        choices=['none', 'light', 'medium', 'heavy', 'aerial', 'custom'],
        help='Augmentation strategy to use during training'
    )
    parser.add_argument(
        '--list-augmentations',
        action='store_true',
        help='List all available augmentation strategies and exit'
    )
    
    # Data arguments
    
    # Model arguments
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        default=True,
        help='Use pretrained Bird Detector weights'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_false',
        dest='use_pretrained',
        help='Train from scratch without pretrained weights'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--score-thresh',
        type=float,
        default=0.4,
        help='Score threshold for predictions'
    )
    parser.add_argument(
        '--nms-thresh',
        type=float,
        default=0.15,
        help='NMS threshold for predictions'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.4,
        help='IoU threshold for evaluation'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained model and results'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='deepforest_finetuned.pt',
        help='Name for saved model file'
    )
    
    # Debug arguments
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run a quick test with minimal data (for debugging)'
    )
    
    args = parser.parse_args()
    main_pipeline(args)
