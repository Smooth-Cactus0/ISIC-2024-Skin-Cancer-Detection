"""
Training script for ConvNeXt on ISIC 2024.

ConvNeXt complements ViT in the ensemble:
- ViT: Global patterns via self-attention
- ConvNeXt: Local patterns via convolutions

This script is nearly identical to train_vit.py but uses ConvNeXt models.
"""

import sys
sys.path.append('..')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedGroupKFold

# Custom modules
from src.models.convnext_classifier import ConvNeXtClassifier, FocalLoss, CONVNEXT_CONFIGS
from src.data.dataset import ISICDataset, BalancedBatchSampler
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.utils.metrics import partial_auc_above_tpr, evaluate_binary_classification


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ConvNeXt for ISIC 2024')

    # Data
    parser.add_argument('--data_dir', type=str, default='../isic-2024-challenge',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='../outputs/convnext',
                       help='Output directory for models and logs')

    # Model
    parser.add_argument('--model_config', type=str, default='convnextv2_nano',
                       choices=list(CONVNEXT_CONFIGS.keys()),
                       help='Model configuration')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                       help='Stochastic depth rate (regularization)')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                       help='Number of epochs to freeze backbone')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                       help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma')

    # Augmentation
    parser.add_argument('--augmentation', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Augmentation strength')

    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--fold', type=int, default=None,
                       help='Train specific fold')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--use_sample', action='store_true',
                       help='Use small sample for testing')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    accumulation_steps: int = 1
) -> float:
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
                loss = criterion(logits, targets)
                loss = loss / accumulation_steps
        else:
            logits = model(images).squeeze()
            loss = criterion(logits, targets)
            loss = loss / accumulation_steps

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
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, total=len(dataloader), desc='Validation')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device).float()

        logits = model(images).squeeze()
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        running_loss += loss.item()
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        pbar.set_postfix({'loss': running_loss / len(dataloader)})

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return running_loss / len(dataloader), all_preds, all_targets


def train_fold(
    fold: int,
    df_train: pd.DataFrame,
    config: dict,
    args: argparse.Namespace
) -> dict:
    """Train single fold."""
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}/{args.n_folds}")
    print(f"{'='*80}")

    # Prepare data
    train_df = df_train[df_train['fold'] != fold].reset_index(drop=True)
    val_df = df_train[df_train['fold'] == fold].reset_index(drop=True)

    print(f"Train: {len(train_df):,} samples ({train_df['target'].sum():,} positive)")
    print(f"Val:   {len(val_df):,} samples ({val_df['target'].sum():,} positive)")

    # Create datasets
    image_dir = Path(args.data_dir) / 'train-image' / 'image'
    train_transform = get_train_transforms(config['image_size'], args.augmentation)
    val_transform = get_val_transforms(config['image_size'])

    train_dataset = ISICDataset(train_df, image_dir, transform=train_transform)
    val_dataset = ISICDataset(val_df, image_dir, transform=val_transform)

    # Dataloaders
    train_sampler = BalancedBatchSampler(
        train_df['target'].values,
        batch_size=args.batch_size,
        positive_ratio=0.5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = ConvNeXtClassifier(
        model_name=config['model_name'],
        pretrained=True,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        freeze_epochs=args.freeze_epochs
    ).to(device)

    print(f"\nModel: {model.model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': model.head.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    scaler = GradScaler() if args.fp16 else None

    # Training loop
    best_pauc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    fold_results = {
        'fold': fold,
        'train_losses': [],
        'val_losses': [],
        'val_aucs': [],
        'val_paucs': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        model.update_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, args.accumulation_steps
        )

        val_loss, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_targets, val_preds)
        val_pauc = partial_auc_above_tpr(val_targets, val_preds, min_tpr=0.8)

        fold_results['train_losses'].append(train_loss)
        fold_results['val_losses'].append(val_loss)
        fold_results['val_aucs'].append(val_auc)
        fold_results['val_paucs'].append(val_pauc)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")
        print(f"Val pAUC:   {val_pauc:.4f} ⭐")

        scheduler.step()

        if val_pauc > best_pauc:
            best_pauc = val_pauc
            best_epoch = epoch
            patience_counter = 0

            checkpoint_dir = Path(args.output_dir) / f'fold_{fold}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_pauc': best_pauc,
                'config': config,
                'args': vars(args)
            }, checkpoint_dir / 'best_model.pth')

            print(f"✓ Saved best model (pAUC: {best_pauc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    fold_results['best_epoch'] = best_epoch
    fold_results['best_pauc'] = best_pauc

    print(f"\nFold {fold} complete - Best pAUC: {best_pauc:.4f} @ epoch {best_epoch + 1}")

    return fold_results


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    config = CONVNEXT_CONFIGS[args.model_config]

    print("="*80)
    print("ISIC 2024 - ConvNeXt Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Image size: {config['image_size']}")
    print(f"  Feature dim: {config['feature_dim']}")
    print(f"  {config['description']}")
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} (head), {args.lr_backbone} (backbone)")
    print(f"  Drop path rate: {args.drop_path_rate}")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Mixed precision: {args.fp16}")

    # Load data
    data_dir = Path(args.data_dir)
    df_train = pd.read_csv(data_dir / 'train-metadata.csv')

    if args.use_sample:
        print(f"\n⚠️  Using sample of 50,000 for quick testing")
        df_train = df_train.sample(50000, random_state=args.seed).reset_index(drop=True)

    print(f"\nData: {len(df_train):,} samples")

    # Create CV folds
    if 'fold' not in df_train.columns:
        df_train['fold'] = -1
        sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        for fold, (_, val_idx) in enumerate(
            sgkf.split(df_train, df_train['target'], df_train['patient_id'])
        ):
            df_train.loc[val_idx, 'fold'] = fold

    # Train folds
    all_results = []

    if args.fold is not None:
        results = train_fold(args.fold, df_train, config, args)
        all_results.append(results)
    else:
        for fold in range(args.n_folds):
            results = train_fold(fold, df_train, config, args)
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

    print(f"\n✓ Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
