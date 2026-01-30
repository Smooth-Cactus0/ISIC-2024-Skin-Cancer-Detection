"""
Extract embeddings from trained ViT models for GBDT ensemble.

This script loads trained ViT models and extracts feature embeddings
for all train/test images. These embeddings will be concatenated with
tabular features for the final multi-modal GBDT ensemble.
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
from torch.utils.data import DataLoader

from src.models.vit_classifier import ViTClassifier, MODEL_CONFIGS
from src.data.dataset import ISICDataset
from src.data.augmentations import get_val_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract ViT embeddings')

    parser.add_argument('--data_dir', type=str, default='../isic-2024-challenge',
                       help='Path to data directory')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='../outputs/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--model_config', type=str, default='vit_tiny',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model configuration')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')

    return parser.parse_args()


@torch.no_grad()
def extract_embeddings_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Extract embeddings from a single model.

    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device

    Returns:
        Tuple of (embeddings, predictions, targets)
    """
    model.eval()

    all_embeddings = []
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc='Extracting embeddings')

    for images, targets in pbar:
        images = images.to(device)

        # Extract features
        features = model.extract_features(images)

        # Get predictions
        logits = model(images).squeeze()
        probs = torch.sigmoid(logits)

        all_embeddings.append(features.cpu().numpy())
        all_predictions.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return embeddings, predictions, targets


def main():
    """Main extraction function."""
    args = parse_args()

    config = MODEL_CONFIGS[args.model_config]

    print("="*80)
    print("ViT Embedding Extraction")
    print("="*80)
    print(f"\nModel: {config['model_name']}")
    print(f"Feature dim: {config['feature_dim']}")
    print(f"Model directory: {args.model_dir}")

    # Load data
    data_dir = Path(args.data_dir)
    df_train = pd.read_csv(data_dir / 'train-metadata.csv')
    df_test = pd.read_csv(data_dir / 'test-metadata.csv')

    print(f"\nTrain samples: {len(df_train):,}")
    print(f"Test samples: {len(df_test):,}")

    # Prepare datasets
    image_dir = data_dir / 'train-image' / 'image'
    transform = get_val_transforms(config['image_size'])

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Storage for all fold embeddings
    train_embeddings_all = np.zeros((len(df_train), config['feature_dim'] * args.n_folds))
    train_predictions_all = np.zeros((len(df_train), args.n_folds))
    test_embeddings_all = np.zeros((len(df_test), config['feature_dim'] * args.n_folds))
    test_predictions_all = np.zeros((len(df_test), args.n_folds))

    # Process each fold
    for fold in range(args.n_folds):
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold}")
        print(f"{'='*80}")

        # Load model
        model_path = Path(args.model_dir) / f'fold_{fold}' / 'best_model.pth'

        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            print(f"   Skipping fold {fold}")
            continue

        checkpoint = torch.load(model_path, map_location=device)

        model = ViTClassifier(
            model_name=config['model_name'],
            pretrained=False  # Loading trained weights
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✓ Loaded model (pAUC: {checkpoint['best_pauc']:.4f})")

        # Extract train embeddings (OOF for this fold)
        val_idx = df_train['fold'] == fold
        val_df = df_train[val_idx].reset_index(drop=True)

        val_dataset = ISICDataset(val_df, image_dir, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        embeddings, predictions, targets = extract_embeddings_from_model(
            model, val_loader, device
        )

        # Store OOF embeddings
        start_idx = fold * config['feature_dim']
        end_idx = (fold + 1) * config['feature_dim']
        train_embeddings_all[val_idx, start_idx:end_idx] = embeddings
        train_predictions_all[val_idx, fold] = predictions

        print(f"  OOF embeddings: {embeddings.shape}")

        # Extract test embeddings
        test_dataset = ISICDataset(df_test, image_dir, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        test_embeddings, test_preds, _ = extract_embeddings_from_model(
            model, test_loader, device
        )

        # Store test embeddings
        test_embeddings_all[:, start_idx:end_idx] = test_embeddings
        test_predictions_all[:, fold] = test_preds

        print(f"  Test embeddings: {test_embeddings.shape}")

    # Average test predictions across folds
    test_predictions_mean = test_predictions_all.mean(axis=1)

    # Save embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training set
    np.save(output_dir / 'train_embeddings.npy', train_embeddings_all)
    np.save(output_dir / 'train_predictions.npy', train_predictions_all.mean(axis=1))

    # Test set
    np.save(output_dir / 'test_embeddings.npy', test_embeddings_all)
    np.save(output_dir / 'test_predictions.npy', test_predictions_mean)

    # Save with metadata
    train_df_with_emb = df_train.copy()
    for i in range(train_embeddings_all.shape[1]):
        train_df_with_emb[f'vit_emb_{i}'] = train_embeddings_all[:, i]
    train_df_with_emb['vit_pred'] = train_predictions_all.mean(axis=1)
    train_df_with_emb.to_parquet(output_dir / 'train_with_embeddings.parquet', index=False)

    test_df_with_emb = df_test.copy()
    for i in range(test_embeddings_all.shape[1]):
        test_df_with_emb[f'vit_emb_{i}'] = test_embeddings_all[:, i]
    test_df_with_emb['vit_pred'] = test_predictions_mean
    test_df_with_emb.to_parquet(output_dir / 'test_with_embeddings.parquet', index=False)

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - train_embeddings.npy: {train_embeddings_all.shape}")
    print(f"  - train_predictions.npy: {train_predictions_all.shape}")
    print(f"  - test_embeddings.npy: {test_embeddings_all.shape}")
    print(f"  - test_predictions.npy: {test_predictions_mean.shape}")
    print(f"  - train_with_embeddings.parquet")
    print(f"  - test_with_embeddings.parquet")

    # Evaluate OOF predictions
    from src.utils.metrics import partial_auc_above_tpr
    from sklearn.metrics import roc_auc_score

    oof_auc = roc_auc_score(df_train['target'], train_predictions_all.mean(axis=1))
    oof_pauc = partial_auc_above_tpr(df_train['target'], train_predictions_all.mean(axis=1), min_tpr=0.8)

    print(f"\n{'='*80}")
    print("ViT MODEL PERFORMANCE (OOF)")
    print(f"{'='*80}")
    print(f"AUC:         {oof_auc:.4f}")
    print(f"pAUC@80TPR:  {oof_pauc:.4f} ⭐")


if __name__ == '__main__':
    main()
