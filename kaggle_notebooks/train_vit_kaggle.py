"""
ISIC 2024 - Vision Transformer Training (Kaggle Notebook)

This is a self-contained script for Kaggle GPU training.
All dependencies are included inline (no external imports from src/).

Usage in Kaggle:
1. Create new notebook
2. Copy this entire file into a code cell
3. Run the cell
4. Model will train and save checkpoints

Expected runtime: 8-12 hours on Kaggle P100 GPU
"""

# ============================================================================
# DEPENDENCIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

# Install required packages (run once)
import subprocess
import sys

def install_packages():
    """Install required packages in Kaggle environment."""
    packages = ['timm', 'albumentations']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

# Uncomment to install (run once at start of notebook):
# install_packages()

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration."""
    # Data
    data_dir = '/kaggle/input/isic-2024-challenge'  # Kaggle data path
    output_dir = '/kaggle/working/outputs'

    # Model
    model_name = 'eva02_base_patch14_224.mim_in22k'  # Best performance
    # Alternative: 'vit_tiny_patch16_224.augreg_in21k' for quick test
    image_size = 224
    dropout = 0.3

    # Training
    n_folds = 5
    fold = None  # None = train all folds
    epochs = 20
    batch_size = 24
    accumulation_steps = 3  # Effective batch = 72
    lr = 5e-4
    lr_backbone = 1e-5
    weight_decay = 1e-4

    # Loss
    focal_alpha = 0.25
    focal_gamma = 2.0

    # Augmentation
    augmentation_strength = 'medium'

    # System
    num_workers = 2
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = True

    # For quick testing
    use_sample = False
    sample_size = 50000


# ============================================================================
# METRICS
# ============================================================================

def partial_auc_above_tpr(y_true, y_pred, min_tpr=0.8):
    """
    Calculate partial AUC above minimum TPR threshold.
    This is the official ISIC 2024 competition metric.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find region where TPR >= min_tpr
    valid_idx = tpr >= min_tpr

    if not np.any(valid_idx):
        return 0.0

    valid_tpr = tpr[valid_idx]
    valid_fpr = fpr[valid_idx]

    # Sort by FPR
    sort_idx = np.argsort(valid_fpr)
    valid_fpr = valid_fpr[sort_idx]
    valid_tpr = valid_tpr[sort_idx]

    # Calculate area
    from sklearn.metrics import auc
    pauc = auc(valid_fpr, valid_tpr)

    # Normalize
    max_fpr = valid_fpr[-1]
    min_fpr = valid_fpr[0]
    max_tpr = valid_tpr[-1]

    max_possible_area = (max_fpr - min_fpr) * (max_tpr - min_tpr)

    if max_possible_area > 0:
        normalized_pauc = pauc / max_possible_area
    else:
        normalized_pauc = 0.0

    return normalized_pauc


# ============================================================================
# MODEL
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


class ViTClassifier(nn.Module):
    """Vision Transformer classifier."""

    def __init__(self, model_name, pretrained=True, dropout=0.0):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        self.feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def extract_features(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return features


# ============================================================================
# DATA
# ============================================================================

def get_train_transforms(image_size=224):
    """Training augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size=224):
    """Validation transforms."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class ISICDataset(Dataset):
    """ISIC 2024 Dataset."""

    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row['isic_id']

        # Load image
        img_path = self.image_dir / f"{isic_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']

        target = row.get('target', -1)
        return img, target


class BalancedBatchSampler:
    """Balanced sampling for extreme class imbalance."""

    def __init__(self, targets, batch_size, positive_ratio=0.5):
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio

        self.positive_indices = np.where(targets == 1)[0]
        self.negative_indices = np.where(targets == 0)[0]

        self.n_positive_per_batch = int(batch_size * positive_ratio)
        self.n_negative_per_batch = batch_size - self.n_positive_per_batch

        self.n_batches = len(self.positive_indices) // self.n_positive_per_batch

    def __iter__(self):
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        for i in range(self.n_batches):
            pos_batch = self.positive_indices[
                i * self.n_positive_per_batch:(i + 1) * self.n_positive_per_batch
            ]
            neg_batch = self.negative_indices[
                i * self.n_negative_per_batch:(i + 1) * self.n_negative_per_batch
            ]

            batch_indices = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch_indices)

            yield batch_indices.tolist()

    def __len__(self):
        return self.n_batches


# ============================================================================
# TRAINING
# ============================================================================

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, accumulation_steps):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training')

    for batch_idx, (images, targets) in pbar:
        images = images.to(device)
        targets = targets.to(device).float()

        if scaler is not None:
            with autocast():
                logits = model(images).squeeze()
                loss = criterion(logits, targets) / accumulation_steps
        else:
            logits = model(images).squeeze()
            loss = criterion(logits, targets) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Validation')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device).float()

        logits = model(images).squeeze()
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        running_loss += loss.item()
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return running_loss / len(dataloader), all_preds, all_targets


def train_fold(fold, df_train, config):
    """Train single fold."""
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{config.n_folds}")
    print(f"{'='*80}")

    # Prepare data
    train_df = df_train[df_train['fold'] != fold].reset_index(drop=True)
    val_df = df_train[df_train['fold'] == fold].reset_index(drop=True)

    print(f"Train: {len(train_df):,} samples ({train_df['target'].sum():,} positive)")
    print(f"Val:   {len(val_df):,} samples ({val_df['target'].sum():,} positive)")

    # Datasets
    image_dir = Path(config.data_dir) / 'train-image/image'
    train_dataset = ISICDataset(train_df, image_dir, get_train_transforms(config.image_size))
    val_dataset = ISICDataset(val_df, image_dir, get_val_transforms(config.image_size))

    # Dataloaders
    train_sampler = BalancedBatchSampler(
        train_df['target'].values,
        batch_size=config.batch_size,
        positive_ratio=0.5
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        drop_last=True,
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

    # Model
    device = torch.device(config.device)
    model = ViTClassifier(
        config.model_name,
        pretrained=True,
        dropout=config.dropout
    ).to(device)

    print(f"\nModel: {config.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config.lr_backbone},
        {'params': model.head.parameters(), 'lr': config.lr}
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    scaler = GradScaler() if config.fp16 else None

    # Training loop
    best_pauc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, config.accumulation_steps
        )

        val_loss, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        val_auc = roc_auc_score(val_targets, val_preds)
        val_pauc = partial_auc_above_tpr(val_targets, val_preds, min_tpr=0.8)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")
        print(f"Val pAUC:   {val_pauc:.4f} ⭐")

        scheduler.step()

        if val_pauc > best_pauc:
            best_pauc = val_pauc
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            checkpoint_dir = Path(config.output_dir) / f'fold_{fold}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_pauc': best_pauc
            }, checkpoint_dir / 'best_model.pth')

            print(f"✓ Saved best model (pAUC: {best_pauc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nFold {fold} complete - Best pAUC: {best_pauc:.4f} @ epoch {best_epoch + 1}")

    return {'fold': fold, 'best_pauc': best_pauc, 'best_epoch': best_epoch}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function."""
    config = Config()
    set_seed(config.seed)

    print("="*80)
    print("ISIC 2024 - Vision Transformer Training (Kaggle)")
    print("="*80)
    print(f"\nModel: {config.model_name}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size} (effective: {config.batch_size * config.accumulation_steps})")
    print(f"Device: {config.device}")
    print(f"FP16: {config.fp16}")

    # Load data
    df_train = pd.read_csv(Path(config.data_dir) / 'train-metadata.csv')

    if config.use_sample:
        print(f"\n⚠️  Using sample of {config.sample_size}")
        df_train = df_train.sample(config.sample_size, random_state=config.seed).reset_index(drop=True)

    print(f"\nData: {len(df_train):,} samples")

    # Create folds
    if 'fold' not in df_train.columns:
        df_train['fold'] = -1
        sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

        for fold, (_, val_idx) in enumerate(
            sgkf.split(df_train, df_train['target'], df_train['patient_id'])
        ):
            df_train.loc[val_idx, 'fold'] = fold

    # Train folds
    all_results = []

    if config.fold is not None:
        results = train_fold(config.fold, df_train, config)
        all_results.append(results)
    else:
        for fold in range(config.n_folds):
            results = train_fold(fold, df_train, config)
            all_results.append(results)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    for res in all_results:
        print(f"\nFold {res['fold']}: Best pAUC = {res['best_pauc']:.4f} @ epoch {res['best_epoch'] + 1}")

    if len(all_results) > 1:
        mean_pauc = np.mean([r['best_pauc'] for r in all_results])
        std_pauc = np.std([r['best_pauc'] for r in all_results])
        print(f"\nMean pAUC: {mean_pauc:.4f} ± {std_pauc:.4f}")

    print(f"\n✓ Models saved to: {config.output_dir}")
    print(f"\n⚠️  Remember to save /kaggle/working/outputs as a Kaggle dataset!")


# Run training
if __name__ == '__main__':
    main()
