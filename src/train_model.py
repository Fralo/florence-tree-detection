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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrame concatenation.*')

config = load_config()
train_config = config["training"]
model_config = config["model"]
data_config = config["data"]


def load_and_validate_data(train_csv, val_csv):
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


def get_transform(augment):
    """
    Get training augmentations.
    
    Args:
        augment (bool): Whether to apply training augmentations
    """
    if augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    else:
        return A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))


def create_model(config):
    """
    Create and configure DeepForest model with augmentations.
    
    Args:
        config: Dictionary with model configuration
    """
    print("\n" + "="*50)
    print("Creating DeepForest model...")
    print("="*50)

    
    # Instantiate DeepForest with
    model = main.deepforest()

    # Weight initialization
    use_pretrained = config.get("use_pretrained", False)
    base_model_path = config.get("base_model_path", None)
    
    if use_pretrained or (not base_model_path):
        # Use official DeepForest pretrained weights from Hugging Face
        print("Loading pretrained DeepForest weights from Hugging Face")
        model.load_model(model_name="weecology/deepforest-tree", revision="main")
    elif base_model_path:
        # Load weights from specified base model path
        print(f"Loading base model from: {base_model_path}")
        model.model = torch.load(base_model_path, weights_only=False)
    
    # Configure model
    model.config["train"]["csv_file"] = config['train_csv']
    model.config["train"]["root_dir"] = config.get('train_root_dir', os.path.dirname(config['train_csv']))
    
    model.config["validation"]["csv_file"] = config['val_csv']
    model.config["validation"]["root_dir"] = config.get('val_root_dir', os.path.dirname(config['val_csv']))
    
    # Set augmentations
    model.transforms = get_transform
    
    # Training hyperparameters
    model.config["batch_size"] = config.get('batch_size', 4)
    model.config["train"]["epochs"] = config.get('epochs', 20)
    model.config["train"]["lr"] = config.get('learning_rate', 0.0001)
    
    model.config["train"]["scheduler"] = {
        "type": "ReduceLROnPlateau",      # torch.optim.lr_scheduler.ReduceLROnPlateau
        "metric": "val_classification",   # the logged validation metric name
        "params": {
            "mode": "min",                # we want the loss to decrease
            "factor": 0.1,                # LR_new = LR_old * factor
            "patience": 5,                # epochs with no improvement
            "min_lr": 1e-7,               # lower LR bound
            "verbose": True,              # log LR changes
            "threshold": 0.0001,          # Required by DeepForest
            "threshold_mode": "rel",      # Required by DeepForest
            "cooldown": 0,                # Required by DeepForest
            "eps": 1e-8                   # Required by DeepForest
        }
    }
    
    model.config["score_thresh"] = config.get('score_thresh', 0.4)
    model.config["nms_thresh"] = config.get('nms_thresh', 0.15)
    
    model.config["workers"] = config.get('num_workers', 4)
    if model.config["workers"] > 0:
        model.config["train"]["persistent_workers"] = True
        model.config["validation"]["persistent_workers"] = True
    
    print(f"✓ Batch size: {model.config['batch_size']}")
    print(f"✓ Epochs: {model.config['train']['epochs']}")
    print(f"✓ Learning rate: {model.config['train']['lr']}")
    print(f"✓ Score threshold: {model.model.score_thresh}")
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
        save_top_k=1,
        verbose=True
    )
    
    csv_logger = CSVLogger(
        save_dir=str(output_dir),
        name="training_logs",
        version=None
    )
    
    # Setup trainer arguments
    trainer_args = {
        "fast_dev_run": config.get('fast_dev_run', False),
        "max_epochs": config.get('epochs', 20),
        "callbacks": [checkpoint_callback],
        "logger": csv_logger,
    }
    
    # Add GPU/MPS/CPU support
    if torch.cuda.is_available():
        print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
        trainer_args["accelerator"] = "gpu"
        trainer_args["devices"] = 1
    elif torch.backends.mps.is_available():
        print("✓ Training on Apple MPS")
        trainer_args["accelerator"] = "mps"
        trainer_args["devices"] = 1
    else:
        print("✓ Training on CPU")
        trainer_args["accelerator"] = "cpu"
        trainer_args["devices"] = 1
    
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
    model_path = output_dir / f"deepforest_finetuned_{existing_models}.pt"
    
    # Save model state dict
    torch.save(model.model, model_path)
    
    print(f"✓ Model (best weights) saved to {model_path}")
    if config.get('best_model_path'):
        print(f"✓ PyTorch Lightning checkpoint also available at: {config['best_model_path']}")
    
    # Save configuration with augmentation info
    config_path = output_dir / f'config_{existing_models}.txt'
    with open(config_path, 'w') as f:
        f.write("DeepForest Training Configuration\n")
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
        'base_model_path': args.base_model_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'score_thresh': args.score_thresh,
        'nms_thresh': args.nms_thresh,
        'num_workers': args.num_workers,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'iou_threshold': args.iou_threshold,
        'fast_dev_run': args.fast_dev_run,
    }
    
    # Load and validate data
    train_df, val_df = load_and_validate_data(training_annotations, validation_annotations)
    
    model = create_model(config)
    
    # Train model (returns path to best model checkpoint)
    best_model_path = train_model(model, config)
    config['best_model_path'] = best_model_path
    
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
    
    # Model arguments
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        default=False,
        help='Use pretrained DeepForest weights (overrides base model path if set)'
    )
    parser.add_argument(
        '--base-model-path',
        type=str,
        default=None,
        help='Path to a base model to start training from'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
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
        default=1,
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
