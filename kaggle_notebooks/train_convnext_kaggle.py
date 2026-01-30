"""
ISIC 2024 - ConvNeXtV2 Training Script for Kaggle
==================================================

Self-contained training script for ConvNeXt models.
Copy this entire file into a Kaggle notebook cell and run.

Architecture: ConvNeXtV2-Nano (3rd place solution)
Expected pAUC@80TPR: ~0.14-0.15 per fold
Training time: ~6-8 hours on Kaggle P100 GPU
"""

import os
import random
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm
import cv2

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for ConvNeXt training"""

    # Paths (modify for Kaggle environment)
    data_dir = '/kaggle/input/isic-2024-challenge'
    train_csv = f'{data_dir}/train-metadata.csv'
    train_images = f'{data_dir}/train-image/image'
    output_dir = '/kaggle/working/outputs'

    # Model selection (choose one):
    # - 'convnextv2_nano.fcmae_ft_in22k_in1k' (fast, used by 3rd place)
    # - 'convnextv2_tiny.fcmae_ft_in22k_in1k' (larger, slower)
    # - 'convnextv2_base.fcmae_ft_in22k_in1k' (best quality, slowest)
    model_name = 'convnextv2_nano.fcmae_ft_in22k_in1k'

    # Training
    fold = None  # None = train all folds, or 0-4 for specific fold
    epochs = 20
    batch_size = 32
    accumulation_steps = 1  # Effective batch = 32 * 1 = 32

    # Learning rates
    backbone_lr = 1e-5  # Pretrained backbone
    head_lr = 1e-3      # Classification head
    weight_decay = 0.01

    # Regularization
    drop_path_rate = 0.1  # Stochastic depth for ConvNeXt
    dropout = 0.2

    # Loss
    focal_alpha = 0.25
    focal_gamma = 2.0

    # Data
    image_size = 224
    num_workers = 2
    seed = 42

    # Mixed precision
    fp16 = True

    # Quick test mode
    use_sample = False  # Set to True for quick 50k sample test
    sample_size = 50000

    # Cross-validation
    n_folds = 5


# ============================================================================
# Metrics
# ============================================================================

def partial_auc_above_tpr(y_true: np.ndarray, y_pred: np.ndarray, min_tpr: float = 0.8) -> float:
    """
    Calculate partial AUC above minimum TPR (ISIC 2024 competition metric).

    This focuses evaluation on high-sensitivity region critical for screening.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Find region where TPR >= min_tpr
    valid_idx = tpr >= min_tpr

    if not np.any(valid_idx):
        return 0.0

    # Extract valid region
    valid_tpr = tpr[valid_idx]
    valid_fpr = fpr[valid_idx]

    # Normalize FPR to [0, 1] range in this region
    fpr_min, fpr_max = valid_fpr.min(), valid_fpr.max()
    if fpr_max - fpr_min < 1e-7:
        return 0.0

    norm_fpr = (valid_fpr - fpr_min) / (fpr_max - fpr_min)

    # Calculate AUC using trapezoidal rule
    pauc = np.trapz(valid_tpr, norm_fpr)

    return pauc


# ============================================================================
# Data Augmentation
# ============================================================================

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentations - medium intensity"""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation transforms - no augmentation"""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================================
# Dataset
# ============================================================================

class ISICDataset(Dataset):
    """ISIC 2024 dataset for image loading"""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform: Optional[A.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']

        target = int(row['target'])

        return image, target


# ============================================================================
# Balanced Sampling
# ============================================================================

class BalancedBatchSampler(Sampler):
    """
    Sampler that creates balanced batches (50% positive, 50% negative).
    Critical for handling 1000:1 class imbalance.
    """

    def __init__(self, targets: np.ndarray, batch_size: int, positive_ratio: float = 0.5):
        self.positive_indices = np.where(targets == 1)[0]
        self.negative_indices = np.where(targets == 0)[0]
        self.batch_size = batch_size
        self.n_positive = int(batch_size * positive_ratio)
        self.n_negative = batch_size - self.n_positive

        # Calculate batches per epoch based on minority class
        self.n_batches = len(self.positive_indices) // self.n_positive

    def __iter__(self):
        for _ in range(self.n_batches):
            # Sample positive and negative indices
            pos_batch = np.random.choice(self.positive_indices, self.n_positive, replace=False)
            neg_batch = np.random.choice(self.negative_indices, self.n_negative, replace=False)

            # Combine and shuffle
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)

            yield from batch

    def __len__(self) -> int:
        return self.n_batches * self.batch_size


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.
    Focuses training on hard examples by down-weighting easy examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Focal term
        focal_weight = (1 - pt) ** self.gamma

        # Alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        loss = alpha_weight * focal_weight * bce_loss

        return loss.mean()


# ============================================================================
# Model
# ============================================================================

class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXtV2 classifier for skin lesion classification.

    Architecture from 3rd place solution - excels at local texture patterns.
    """

    def __init__(
        self,
        model_name: str = 'convnextv2_nano.fcmae_ft_in22k_in1k',
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        dropout: float = 0.2
    ):
        super().__init__()

        # Load pretrained ConvNeXt backbone (without classifier)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove head
            drop_path_rate=drop_path_rate  # Stochastic depth
        )

        # Get feature dimension
        self.num_features = self.backbone.num_features

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.num_features, 1)

        # Initialize head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)

        # Classify
        features = self.dropout(features)
        logits = self.head(features)

        return logits.squeeze(-1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for ensemble"""
        return self.backbone(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 1,
    fp16: bool = True
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Training')
    for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.float().to(device)

        # Forward pass with mixed precision
        with autocast(enabled=fp16):
            logits = model(images)
            loss = criterion(logits, targets) / accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    fp16: bool = True
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc='Validation')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.float().to(device)

        with autocast(enabled=fp16):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item()

        # Collect predictions
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(loader)
    pauc = partial_auc_above_tpr(all_targets, all_preds, min_tpr=0.8)

    return avg_loss, pauc, all_preds, all_targets


# ============================================================================
# Main Training Loop
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config
) -> Dict:
    """Train a single fold"""

    print(f"\n{'='*80}")
    print(f"Training Fold {fold}")
    print(f"{'='*80}")
    print(f"Train: {len(train_df)} samples ({train_df['target'].sum()} malignant)")
    print(f"Val:   {len(val_df)} samples ({val_df['target'].sum()} malignant)")

    # Create datasets
    train_dataset = ISICDataset(
        train_df,
        config.train_images,
        transform=get_train_transforms(config.image_size)
    )
    val_dataset = ISICDataset(
        val_df,
        config.train_images,
        transform=get_val_transforms(config.image_size)
    )

    # Create dataloaders with balanced sampling
    train_sampler = BalancedBatchSampler(
        train_df['target'].values,
        config.batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNeXtClassifier(
        model_name=config.model_name,
        pretrained=True,
        drop_path_rate=config.drop_path_rate,
        dropout=config.dropout
    ).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config.backbone_lr},
        {'params': model.head.parameters(), 'lr': config.head_lr}
    ], weight_decay=config.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=config.fp16)

    # Training loop
    best_pauc = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_pauc': []}

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config.accumulation_steps, config.fp16
        )

        # Validate
        val_loss, val_pauc, _, _ = validate_epoch(
            model, val_loader, criterion, device, config.fp16
        )

        # Update scheduler
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pauc'].append(val_pauc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val pAUC:   {val_pauc:.4f}")

        # Save best model
        if val_pauc > best_pauc:
            best_pauc = val_pauc
            best_epoch = epoch

            # Save checkpoint
            fold_dir = Path(config.output_dir) / f'fold_{fold}'
            fold_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'best_pauc': best_pauc,
                'epoch': epoch
            }, fold_dir / 'best_model.pth')

            print(f"✓ Saved best model (pAUC: {best_pauc:.4f})")

    print(f"\nFold {fold} Complete!")
    print(f"Best pAUC: {best_pauc:.4f} @ epoch {best_epoch+1}")

    return {
        'fold': fold,
        'best_pauc': best_pauc,
        'best_epoch': best_epoch,
        'history': history
    }


def main():
    """Main training function"""
    config = Config()
    set_seed(config.seed)

    print("="*80)
    print("ISIC 2024 - ConvNeXtV2 Training")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Image size: {config.image_size}")
    print(f"Batch size: {config.batch_size} x {config.accumulation_steps} = {config.batch_size * config.accumulation_steps}")
    print(f"Epochs: {config.epochs}")
    print(f"FP16: {config.fp16}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(config.train_csv)

    if config.use_sample:
        print(f"Using {config.sample_size} sample for quick test")
        df = df.sample(n=min(config.sample_size, len(df)), random_state=config.seed)

    print(f"Total samples: {len(df)}")
    print(f"Malignant: {df['target'].sum()} ({df['target'].mean()*100:.2f}%)")

    # Create folds
    print("\nCreating cross-validation splits...")
    skf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    df['fold'] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df['target'], df['patient_id'])):
        df.loc[val_idx, 'fold'] = fold

    # Train folds
    results = []
    folds_to_train = [config.fold] if config.fold is not None else range(config.n_folds)

    for fold in folds_to_train:
        train_df = df[df['fold'] != fold].copy()
        val_df = df[df['fold'] == fold].copy()

        fold_results = train_fold(fold, train_df, val_df, config)
        results.append(fold_results)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    for result in results:
        print(f"Fold {result['fold']}: Best pAUC = {result['best_pauc']:.4f} @ epoch {result['best_epoch']+1}")

    if len(results) > 1:
        mean_pauc = np.mean([r['best_pauc'] for r in results])
        std_pauc = np.std([r['best_pauc'] for r in results])
        print(f"\nMean pAUC: {mean_pauc:.4f} ± {std_pauc:.4f}")

    print(f"\nOutputs saved to: {config.output_dir}")


if __name__ == '__main__':
    main()
