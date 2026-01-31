"""
Dataset classes for ISIC 2024 Skin Cancer Detection.

Supports both individual JPEG files and HDF5 format for efficient loading.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import h5py
from typing import Optional, Callable, Tuple, Union

import torch
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2024 images.

    Supports:
    - Loading from individual JPEG files
    - Loading from HDF5 file (faster I/O)
    - Optional metadata integration
    - Configurable transforms/augmentations
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_source: Union[str, Path],
        transform: Optional[Callable] = None,
        use_hdf5: bool = False,
        return_metadata: bool = False,
        metadata_cols: Optional[list] = None
    ):
        """
        Initialize ISIC dataset.

        Args:
            df: DataFrame with at minimum 'isic_id' and optionally 'target'
            image_source: Path to image directory or HDF5 file
            transform: Albumentations or torchvision transforms
            use_hdf5: If True, load from HDF5; else from individual JPEGs
            return_metadata: If True, return (image, target, metadata) else (image, target)
            metadata_cols: List of metadata columns to return (if return_metadata=True)
        """
        self.df = df.reset_index(drop=True)
        self.image_source = Path(image_source)
        self.transform = transform
        self.use_hdf5 = use_hdf5
        self.return_metadata = return_metadata
        self.metadata_cols = metadata_cols or []

        # Open HDF5 file if using that format
        self.hdf5_file = None
        if self.use_hdf5:
            if self.image_source.suffix == '.hdf5' and self.image_source.exists():
                # HDF5 file should be opened in __getitem__ for multiprocessing compatibility
                pass
            else:
                raise FileNotFoundError(f"HDF5 file not found: {self.image_source}")

        # Verify at least one image exists
        if not self.use_hdf5:
            sample_id = self.df.iloc[0]['isic_id']
            sample_path = self.image_source / f"{sample_id}.jpg"
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"Sample image not found: {sample_path}\n"
                    f"Please check image_source path: {self.image_source}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample from the dataset.

        Returns:
            If return_metadata=False: (image, target)
            If return_metadata=True: (image, target, metadata_dict)
        """
        row = self.df.iloc[idx]
        isic_id = row['isic_id']

        # Load image
        if self.use_hdf5:
            image = self._load_from_hdf5(isic_id)
        else:
            image = self._load_from_jpeg(isic_id)

        # Apply transforms
        if self.transform is not None:
            # Check if it's albumentations (has __call__ with image parameter)
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except (TypeError, KeyError):
                # Assume torchvision transform
                image = self.transform(image)

        # Get target (if available)
        target = row.get('target', -1)  # -1 for test set without labels

        if not self.return_metadata:
            return image, target
        else:
            # Return metadata as dict
            metadata = {col: row[col] for col in self.metadata_cols if col in row}
            return image, target, metadata

    def _load_from_jpeg(self, isic_id: str) -> np.ndarray:
        """Load image from JPEG file."""
        img_path = self.image_source / f"{isic_id}.jpg"
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def _load_from_hdf5(self, isic_id: str) -> np.ndarray:
        """Load image from HDF5 file."""
        # Open file per access for multiprocessing compatibility
        with h5py.File(self.image_source, 'r') as hf:
            img = hf[isic_id][:]
        return img


class ISICTabularDataset(Dataset):
    """
    Dataset for tabular features only (for GBDT training).

    Returns features and targets as numpy arrays.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'target'
    ):
        """
        Initialize tabular dataset.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Extract features and targets as arrays
        self.X = self.df[feature_cols].values.astype(np.float32)

        if target_col in self.df.columns:
            self.y = self.df[target_col].values.astype(np.float32)
        else:
            self.y = np.full(len(self.df), -1, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self.X[idx], self.y[idx]


class BalancedBatchSampler:
    """
    Batch sampler that ensures each batch has equal numbers of positive and negative samples.

    Critical for handling extreme class imbalance (~1000:1 in ISIC 2024).
    """

    def __init__(
        self,
        targets: np.ndarray,
        batch_size: int,
        positive_ratio: float = 0.5
    ):
        """
        Initialize balanced batch sampler.

        Args:
            targets: Array of binary targets (0/1)
            batch_size: Desired batch size
            positive_ratio: Ratio of positive samples in each batch (default: 0.5)
        """
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio

        # Split indices by class
        self.positive_indices = np.where(targets == 1)[0]
        self.negative_indices = np.where(targets == 0)[0]

        # Calculate samples per batch
        self.n_positive_per_batch = int(batch_size * positive_ratio)
        self.n_negative_per_batch = batch_size - self.n_positive_per_batch

        # Calculate number of batches
        # Limited by the minority class
        self.n_batches = len(self.positive_indices) // self.n_positive_per_batch

    def __iter__(self):
        """Generate balanced batches."""
        # Shuffle indices
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        for i in range(self.n_batches):
            # Sample positive and negative indices
            pos_batch = self.positive_indices[
                i * self.n_positive_per_batch:(i + 1) * self.n_positive_per_batch
            ]
            neg_batch = self.negative_indices[
                i * self.n_negative_per_batch:(i + 1) * self.n_negative_per_batch
            ]

            # Combine and shuffle within batch
            batch_indices = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch_indices)

            yield from batch_indices.tolist()

    def __len__(self) -> int:
        return self.n_batches * self.batch_size


def create_fold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    patient_col: str = 'patient_id',
    target_col: str = 'target',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create stratified group k-fold splits.

    Args:
        df: DataFrame with data
        n_folds: Number of folds
        patient_col: Column name for patient grouping
        target_col: Column name for target variable
        random_state: Random seed

    Returns:
        DataFrame with added 'fold' column
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df['fold'] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, df[target_col], df[patient_col])
    ):
        df.loc[val_idx, 'fold'] = fold

    return df


if __name__ == "__main__":
    # Test dataset creation
    print("Testing ISIC dataset classes...\n")

    # Create dummy data
    dummy_df = pd.DataFrame({
        'isic_id': [f'ISIC_{i:07d}' for i in range(100)],
        'target': np.random.randint(0, 2, 100),
        'patient_id': [f'P{i//10:03d}' for i in range(100)],
        'age_approx': np.random.randint(20, 80, 100)
    })

    print(f"Created dummy data: {len(dummy_df)} samples")
    print(f"Class distribution: {dummy_df['target'].value_counts().to_dict()}")

    # Test tabular dataset
    feature_cols = ['age_approx']
    tabular_dataset = ISICTabularDataset(dummy_df, feature_cols)
    print(f"\n✓ Tabular dataset: {len(tabular_dataset)} samples")

    # Test balanced sampler
    sampler = BalancedBatchSampler(dummy_df['target'].values, batch_size=16)
    print(f"✓ Balanced sampler: {len(sampler)} batches")

    # Test fold creation
    df_with_folds = create_fold_splits(dummy_df, n_folds=5)
    print(f"✓ Created 5-fold splits:")
    print(df_with_folds['fold'].value_counts().sort_index())

    # Verify no patient leakage
    for fold1 in range(5):
        for fold2 in range(fold1 + 1, 5):
            patients_f1 = set(df_with_folds[df_with_folds['fold'] == fold1]['patient_id'])
            patients_f2 = set(df_with_folds[df_with_folds['fold'] == fold2]['patient_id'])
            assert len(patients_f1 & patients_f2) == 0, f"Leakage between fold {fold1} and {fold2}"

    print("✓ No patient leakage detected")
    print("\n✓ Dataset modules tested successfully")
