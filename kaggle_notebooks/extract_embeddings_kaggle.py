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
- test_embeddings.npy: (N_test, embedding_dim) embeddings  [only during submission]
- test_predictions.npy: (N_test,) averaged predictions      [only during submission]
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import cv2
from sklearn.model_selection import StratifiedGroupKFold

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

    # Model weights directories - upload trained models as Kaggle datasets
    # Each model type points to its own uploaded dataset
    model_dirs = {
        'convnext': '/kaggle/input/isic-train-convnext',
        'eva02': '/kaggle/input/isic-train-vit',       # UPDATE after ViT training
        'vit': '/kaggle/input/isic-train-vit',          # UPDATE after ViT training
    }

    # Model architecture - CHANGE THIS to switch models
    # Options: 'convnext', 'eva02', 'vit'
    model_type = 'convnext'

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

    # Cross-validation (must match training scripts)
    n_folds = 5
    seed = 42


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

        if image is None:
            # Return a black placeholder image if file is missing
            image = np.zeros((self.transform.transforms[0].height,
                              self.transform.transforms[0].width, 3), dtype=np.uint8)
        else:
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

    def predict_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get predictions from pre-computed embeddings (avoids double forward pass)"""
        features = self.dropout(embeddings)
        logits = self.head(features)
        return logits.squeeze(-1)


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

    def predict_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get predictions from pre-computed embeddings (avoids double forward pass)"""
        features = self.dropout(embeddings)
        logits = self.head(features)
        return logits.squeeze(-1)


def create_model(model_name: str, model_type: str) -> nn.Module:
    """Create model based on type"""
    if model_type in ['vit', 'eva02']:
        return ViTClassifier(model_name)
    elif model_type == 'convnext':
        return ConvNeXtClassifier(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# Checkpoint Loading
# ============================================================================

def find_checkpoint(model_dir: str, fold: int) -> Optional[Path]:
    """Find checkpoint file, trying multiple directory layouts."""
    for subdir in ['', 'outputs']:
        candidate = Path(model_dir) / subdir / f'fold_{fold}' / 'best_model.pth'
        if candidate.exists():
            return candidate
    return None


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
    Uses single forward pass through backbone, then computes predictions
    from embeddings to avoid redundant computation.

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

        with torch.amp.autocast('cuda', enabled=fp16 and device.type == 'cuda'):
            # Single forward pass through backbone
            embeddings = model.get_embeddings(images)
            # Reuse embeddings for predictions (no second backbone pass)
            logits = model.predict_from_embeddings(embeddings)
            predictions = torch.sigmoid(logits)

        all_embeddings.append(embeddings.float().cpu().numpy())
        all_predictions.append(predictions.float().cpu().numpy())
        all_ids.extend(ids)

    # Concatenate all batches
    embeddings = np.concatenate(all_embeddings, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    return embeddings, predictions, all_ids


def extract_train_embeddings(
    df: pd.DataFrame,
    config: Config
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Extract train embeddings using out-of-fold predictions.

    For each sample, use the model trained on folds where it was NOT in training set.
    This provides unbiased predictions for ensemble training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transforms(config.image_size)

    # Initialize arrays
    n_samples = len(df)
    all_embeddings = None
    all_predictions = np.zeros(n_samples)
    folds_loaded = 0

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

        checkpoint_path = find_checkpoint(config.model_dir, fold)
        if checkpoint_path is None:
            print(f"WARNING: Checkpoint not found for fold {fold} in {config.model_dir}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded: {checkpoint_path.name} (pAUC: {checkpoint['best_pauc']:.4f})")

        # Extract embeddings and predictions
        embeddings, predictions, ids = extract_embeddings_from_loader(
            model, loader, device, config.fp16
        )

        # Initialize embedding array on first successful fold
        if all_embeddings is None:
            embedding_dim = embeddings.shape[1]
            all_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)
            print(f"Embedding dimension: {embedding_dim}")

        # Store out-of-fold predictions
        val_indices = df[df['fold'] == fold].index
        all_embeddings[val_indices] = embeddings
        all_predictions[val_indices] = predictions
        folds_loaded += 1

        # Clean up
        del model
        torch.cuda.empty_cache()

    if folds_loaded == 0:
        print("\nERROR: No checkpoints were loaded! Check model_dir paths.")
        return None, all_predictions

    print(f"\nSuccessfully extracted from {folds_loaded}/{config.n_folds} folds.")
    return all_embeddings, all_predictions


def extract_test_embeddings(
    df: pd.DataFrame,
    config: Config
) -> Tuple[Optional[np.ndarray], np.ndarray]:
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
    all_embeddings_sum = None
    all_predictions_sum = np.zeros(n_samples)
    folds_loaded = 0

    print("\nExtracting test embeddings (averaging all folds)...")

    for fold in range(config.n_folds):
        print(f"\n--- Fold {fold} ---")

        # Load model
        model = create_model(
            config.model_configs[config.model_type],
            config.model_type
        ).to(device)

        checkpoint_path = find_checkpoint(config.model_dir, fold)
        if checkpoint_path is None:
            print(f"WARNING: Checkpoint not found for fold {fold} in {config.model_dir}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded: {checkpoint_path.name} (pAUC: {checkpoint['best_pauc']:.4f})")

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
        folds_loaded += 1

        # Clean up
        del model
        torch.cuda.empty_cache()

    if folds_loaded == 0:
        print("\nERROR: No checkpoints loaded for test extraction!")
        return None, all_predictions_sum

    # Average across loaded folds
    all_embeddings = all_embeddings_sum / folds_loaded
    all_predictions = all_predictions_sum / folds_loaded

    return all_embeddings, all_predictions


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main extraction function"""
    config = Config()

    # Resolve model directory from dict
    config.model_dir = config.model_dirs[config.model_type]

    print("="*80)
    print("ISIC 2024 - Embedding Extraction")
    print("="*80)
    print(f"Model type: {config.model_type}")
    print(f"Model name: {config.model_configs[config.model_type]}")
    print(f"Model dir: {config.model_dir}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Disable FP16 on CPU (autocast only works with CUDA)
    if device == 'cpu' and config.fp16:
        print("WARNING: FP16 disabled (requires CUDA). Using FP32.")
        print("TIP: Enable GPU accelerator in Kaggle notebook settings.")
        config.fp16 = False

    # Verify model dir exists and show checkpoints
    model_path = Path(config.model_dir)
    if not model_path.exists():
        print(f"\nERROR: Model directory not found: {config.model_dir}")
        print("Make sure you've added the trained model as an input dataset.")
        return

    checkpoints = sorted(model_path.rglob('*.pth'))
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for cp in checkpoints:
        print(f"  {cp}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("\nLoading metadata...")
    train_df = pd.read_csv(config.train_csv, low_memory=False)
    test_df = pd.read_csv(config.test_csv, low_memory=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Create fold column (reproducing exact same splits used during training)
    if 'fold' not in train_df.columns:
        print("\nCreating cross-validation folds (seed=42, same as training)...")
        train_df['fold'] = -1
        sgkf = StratifiedGroupKFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.seed
        )
        for fold, (_, val_idx) in enumerate(
            sgkf.split(train_df, train_df['target'], train_df['patient_id'])
        ):
            train_df.loc[val_idx, 'fold'] = fold

        for f in range(config.n_folds):
            n = (train_df['fold'] == f).sum()
            pos = train_df.loc[train_df['fold'] == f, 'target'].sum()
            print(f"  Fold {f}: {n:,} samples ({pos} malignant)")
    else:
        print("\nUsing existing fold assignments from metadata.")

    # ====================================================================
    # Extract TRAIN embeddings (out-of-fold)
    # ====================================================================
    train_embeddings, train_predictions = extract_train_embeddings(train_df, config)

    if train_embeddings is not None:
        print("\nSaving train outputs...")
        np.save(output_dir / 'train_embeddings.npy', train_embeddings)
        np.save(output_dir / 'train_predictions.npy', train_predictions)
        np.save(output_dir / 'train_ids.npy', train_df['isic_id'].values)
        np.save(output_dir / 'train_targets.npy', train_df['target'].values)
        print(f"  train_embeddings.npy  {train_embeddings.shape}")
        print(f"  train_predictions.npy {train_predictions.shape}")
        print(f"  train_ids.npy         ({len(train_df)},)")
        print(f"  train_targets.npy     ({len(train_df)},)")
    else:
        print("\nERROR: Train embedding extraction failed. No outputs saved.")
        return

    # ====================================================================
    # Extract TEST embeddings (only if test images exist on disk)
    # ====================================================================
    test_image_dir = Path(config.test_images)
    # Check if test images actually exist (they only exist during Kaggle submission)
    test_images_available = (
        test_image_dir.exists()
        and len(test_df) > 0
        and any(test_image_dir.glob('*.jpg'))
    )

    if test_images_available:
        print(f"\nTest images found at {config.test_images}")
        test_embeddings, test_predictions = extract_test_embeddings(test_df, config)

        if test_embeddings is not None:
            print("\nSaving test outputs...")
            np.save(output_dir / 'test_embeddings.npy', test_embeddings)
            np.save(output_dir / 'test_predictions.npy', test_predictions)
            np.save(output_dir / 'test_ids.npy', test_df['isic_id'].values)
            print(f"  test_embeddings.npy  {test_embeddings.shape}")
            print(f"  test_predictions.npy {test_predictions.shape}")
    else:
        print(f"\nSkipping test extraction — test images not found at {config.test_images}")
        print("(This is normal during development. Test images are only available during submission.)")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"\nFiles:")
    for f in sorted(output_dir.glob('*.npy')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    print("\nNext steps:")
    if config.model_type == 'convnext':
        print("1. Change model_type = 'eva02' and run again for ViT embeddings")
    elif config.model_type in ['eva02', 'vit']:
        print("1. Change model_type = 'convnext' and run again for ConvNeXt embeddings")
    print("2. Upload embeddings as Kaggle dataset")
    print("3. Run ensemble_kaggle.py to train final GBDT")


if __name__ == '__main__':
    main()
