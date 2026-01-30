"""
ISIC 2024 - Embedding Extraction Script for Kaggle
===================================================

Extracts feature embeddings from trained ViT/ConvNeXt models for ensemble.
Copy this entire file into a Kaggle notebook cell and run.

Usage:
1. Train ViT and ConvNeXt models first
2. Upload model weights as Kaggle datasets
3. Run this script to extract embeddings
4. Use embeddings for final GBDT ensemble

Expected output:
- train_embeddings.npy: (N_train, embedding_dim) embeddings
- train_predictions.npy: (N_train,) out-of-fold predictions
- test_embeddings.npy: (N_test, embedding_dim) embeddings
- test_predictions.npy: (N_test,) averaged predictions
"""

import os
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import cv2

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for embedding extraction"""

    # Paths - MODIFY THESE for your Kaggle setup
    data_dir = '/kaggle/input/isic-2024-challenge'
    train_csv = f'{data_dir}/train-metadata.csv'
    test_csv = f'{data_dir}/test-metadata.csv'
    train_images = f'{data_dir}/train-image/image'
    test_images = f'{data_dir}/test-image/image'

    # Model weights directory - upload trained models as Kaggle dataset
    model_dir = '/kaggle/input/trained-models'  # MODIFY THIS

    # Model architecture
    # Options: 'vit', 'eva02', 'convnext'
    model_type = 'eva02'  # MODIFY THIS

    # Corresponding model name
    model_configs = {
        'vit': 'vit_base_patch16_224.augreg_in21k',
        'eva02': 'eva02_base_patch14_224.mim_in22k',
        'convnext': 'convnextv2_nano.fcmae_ft_in22k_in1k'
    }

    # Output
    output_dir = '/kaggle/working/embeddings'

    # Inference
    image_size = 224
    batch_size = 64  # Larger batch for inference
    num_workers = 2
    fp16 = True

    # Cross-validation
    n_folds = 5


# ============================================================================
# Data Loading
# ============================================================================

def get_transforms(image_size: int = 224) -> A.Compose:
    """Inference transforms - no augmentation"""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class ISICDataset(Dataset):
    """ISIC 2024 dataset for inference"""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform: A.Compose):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        image = self.transform(image=image)['image']

        return image, row['isic_id']


# ============================================================================
# Model Definitions
# ============================================================================

class ViTClassifier(nn.Module):
    """Vision Transformer classifier"""

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.head(features)
        return logits.squeeze(-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without classification head"""
        return self.backbone(x)


class ConvNeXtClassifier(nn.Module):
    """ConvNeXt classifier"""

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, drop_path_rate=0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.head(features)
        return logits.squeeze(-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without classification head"""
        return self.backbone(x)


def create_model(model_name: str, model_type: str) -> nn.Module:
    """Create model based on type"""
    if model_type in ['vit', 'eva02']:
        return ViTClassifier(model_name)
    elif model_type == 'convnext':
        return ConvNeXtClassifier(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Inference Functions
# ============================================================================

@torch.no_grad()
def extract_embeddings_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool = True
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract embeddings and predictions from a dataloader.

    Returns:
        embeddings: (N, embedding_dim) feature vectors
        predictions: (N,) probability predictions
        ids: (N,) sample IDs
    """
    model.eval()

    all_embeddings = []
    all_predictions = []
    all_ids = []

    for images, ids in tqdm(loader, desc='Extracting'):
        images = images.to(device)

        with autocast(enabled=fp16):
            # Get embeddings
            embeddings = model.get_embeddings(images)

            # Get predictions
            logits = model(images)
            predictions = torch.sigmoid(logits)

        all_embeddings.append(embeddings.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_ids.extend(ids)

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    return embeddings, predictions, all_ids


def extract_train_embeddings(
    df: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract train embeddings using out-of-fold predictions.

    For each sample, use the model trained on folds where it was NOT in training set.
    This provides unbiased predictions for ensemble training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transforms(config.image_size)

    # Initialize arrays
    n_samples = len(df)
    embedding_dim = None  # Will be determined from first batch
    all_embeddings = None
    all_predictions = np.zeros(n_samples)

    print("\nExtracting train embeddings (out-of-fold)...")

    for fold in range(config.n_folds):
        print(f"\n--- Fold {fold} ---")

        # Get validation samples for this fold
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        print(f"Validation samples: {len(val_df)}")

        # Create dataset and loader
        dataset = ISICDataset(val_df, config.train_images, transform)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Load model trained on this fold
        model = create_model(
            config.model_configs[config.model_type],
            config.model_type
        ).to(device)

        checkpoint_path = Path(config.model_dir) / f'fold_{fold}' / 'best_model.pth'
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (pAUC: {checkpoint['best_pauc']:.4f})")

        # Extract embeddings and predictions
        embeddings, predictions, ids = extract_embeddings_from_loader(
            model, loader, device, config.fp16
        )

        # Initialize embedding array on first fold
        if all_embeddings is None:
            embedding_dim = embeddings.shape[1]
            all_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
            print(f"Embedding dimension: {embedding_dim}")

        # Store out-of-fold predictions
        val_indices = df[df['fold'] == fold].index
        all_embeddings[val_indices] = embeddings
        all_predictions[val_indices] = predictions

        # Clean up
        del model
        torch.cuda.empty_cache()

    return all_embeddings, all_predictions


def extract_test_embeddings(
    df: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract test embeddings by averaging predictions from all folds.

    This provides more robust predictions through ensemble of fold models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transforms(config.image_size)

    # Create dataset and loader
    dataset = ISICDataset(df, config.test_images, transform)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Initialize accumulators
    n_samples = len(df)
    embedding_dim = None
    all_embeddings_sum = None
    all_predictions_sum = np.zeros(n_samples)

    print("\nExtracting test embeddings (averaging all folds)...")

    for fold in range(config.n_folds):
        print(f"\n--- Fold {fold} ---")

        # Load model
        model = create_model(
            config.model_configs[config.model_type],
            config.model_type
        ).to(device)

        checkpoint_path = Path(config.model_dir) / f'fold_{fold}' / 'best_model.pth'
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (pAUC: {checkpoint['best_pauc']:.4f})")

        # Extract embeddings and predictions
        embeddings, predictions, _ = extract_embeddings_from_loader(
            model, loader, device, config.fp16
        )

        # Initialize on first fold
        if all_embeddings_sum is None:
            embedding_dim = embeddings.shape[1]
            all_embeddings_sum = np.zeros((n_samples, embedding_dim), dtype=np.float32)
            print(f"Embedding dimension: {embedding_dim}")

        # Accumulate
        all_embeddings_sum += embeddings
        all_predictions_sum += predictions

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Average across folds
    all_embeddings = all_embeddings_sum / config.n_folds
    all_predictions = all_predictions_sum / config.n_folds

    return all_embeddings, all_predictions


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main extraction function"""
    config = Config()

    print("="*80)
    print("ISIC 2024 - Embedding Extraction")
    print("="*80)
    print(f"Model type: {config.model_type}")
    print(f"Model name: {config.model_configs[config.model_type]}")
    print(f"Model dir: {config.model_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("\nLoading metadata...")
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Check for fold column
    if 'fold' not in train_df.columns:
        print("\nWARNING: 'fold' column not found in train metadata!")
        print("You need to add cross-validation folds first.")
        print("Use StratifiedGroupKFold with patient_id as groups.")
        return

    # Extract train embeddings (out-of-fold)
    train_embeddings, train_predictions = extract_train_embeddings(train_df, config)

    # Save train outputs
    print("\nSaving train outputs...")
    np.save(output_dir / 'train_embeddings.npy', train_embeddings)
    np.save(output_dir / 'train_predictions.npy', train_predictions)
    np.save(output_dir / 'train_ids.npy', train_df['isic_id'].values)
    print(f"Saved: train_embeddings.npy {train_embeddings.shape}")
    print(f"Saved: train_predictions.npy {train_predictions.shape}")

    # Extract test embeddings (averaged across folds)
    test_embeddings, test_predictions = extract_test_embeddings(test_df, config)

    # Save test outputs
    print("\nSaving test outputs...")
    np.save(output_dir / 'test_embeddings.npy', test_embeddings)
    np.save(output_dir / 'test_predictions.npy', test_predictions)
    np.save(output_dir / 'test_ids.npy', test_df['isic_id'].values)
    print(f"Saved: test_embeddings.npy {test_embeddings.shape}")
    print(f"Saved: test_predictions.npy {test_predictions.shape}")

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Repeat for other model types (ViT, ConvNeXt)")
    print("2. Combine embeddings with tabular features")
    print("3. Train final GBDT ensemble")


if __name__ == '__main__':
    main()
