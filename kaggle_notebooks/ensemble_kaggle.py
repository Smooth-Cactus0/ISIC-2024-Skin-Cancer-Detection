"""
ISIC 2024 - Final Ensemble Script for Kaggle
=============================================

Combines image embeddings + tabular features into final GBDT ensemble.
Copy this entire file into a Kaggle notebook cell and run.

Pipeline:
1. Load ViT embeddings (768-dim)
2. Load ConvNeXt embeddings (640-dim)
3. Load engineered tabular features (~200-dim)
4. Train LightGBM with 5-fold CV
5. Generate submission file

Expected performance: pAUC@80TPR ~0.16-0.18
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for ensemble training"""

    # Paths - MODIFY THESE for your Kaggle setup
    data_dir = '/kaggle/input/isic-2024-challenge'
    train_csv = f'{data_dir}/train-metadata.csv'
    test_csv = f'{data_dir}/test-metadata.csv'

    # Embedding paths - upload extracted embeddings as Kaggle datasets
    vit_embeddings_dir = '/kaggle/input/vit-embeddings'
    convnext_embeddings_dir = '/kaggle/input/convnext-embeddings'

    # Output
    output_dir = '/kaggle/working'
    submission_file = 'submission.csv'

    # LightGBM parameters (tuned for imbalanced data)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'scale_pos_weight': 1000,  # Adjust for class imbalance
        'verbose': -1,
        'random_state': 42
    }

    num_boost_round = 1000
    early_stopping_rounds = 50

    # Cross-validation
    n_folds = 5
    seed = 42


# ============================================================================
# Feature Engineering
# ============================================================================

def create_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient-normalized features.

    Key insight from top solutions: Features relative to patient's other lesions
    are more informative than absolute values.
    """
    # TBP features to normalize
    tbp_features = [
        'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
        'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext',
        'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
        'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB',
        'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
        'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence',
        'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
        'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
    ]

    # Filter to existing columns
    tbp_features = [col for col in tbp_features if col in df.columns]

    features = df.copy()

    # Patient-level normalization
    for col in tbp_features:
        # Z-score within patient
        patient_mean = df.groupby('patient_id')[col].transform('mean')
        patient_std = df.groupby('patient_id')[col].transform('std')
        features[f'{col}_zscore'] = (df[col] - patient_mean) / (patient_std + 1e-8)

        # Percentile rank within patient
        features[f'{col}_percentile'] = df.groupby('patient_id')[col].rank(pct=True)

        # Deviation from patient median
        patient_median = df.groupby('patient_id')[col].transform('median')
        features[f'{col}_dev_median'] = df[col] - patient_median

    # Patient statistics
    patient_stats = df.groupby('patient_id').agg({
        'isic_id': 'count',  # Number of lesions
        **{col: ['mean', 'std', 'min', 'max'] for col in tbp_features[:5]}  # Top 5 features
    }).reset_index()

    patient_stats.columns = ['patient_id'] + [
        f'patient_{col}_{stat}' for col, stat in patient_stats.columns[1:]
    ]

    features = features.merge(patient_stats, on='patient_id', how='left')

    return features


def create_abcde_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ABCDE clinical criteria features.

    ABCDE = Asymmetry, Border, Color, Diameter, Evolution
    These are dermatological red flags for melanoma.
    """
    features = df.copy()

    # A - Asymmetry (lower symmetry = more asymmetric = more suspicious)
    if 'tbp_lv_symm_2axis' in df.columns:
        features['abcde_asymmetry'] = 1 - df['tbp_lv_symm_2axis']

    # B - Border irregularity (higher norm_border = more irregular)
    if 'tbp_lv_norm_border' in df.columns:
        features['abcde_border'] = df['tbp_lv_norm_border']

    # C - Color variation (higher color_std = more variation)
    if 'tbp_lv_color_std_mean' in df.columns:
        features['abcde_color_variation'] = df['tbp_lv_color_std_mean']

    # Color diversity (angular color variation)
    if 'tbp_lv_radial_color_std_max' in df.columns:
        features['abcde_radial_color'] = df['tbp_lv_radial_color_std_max']

    # D - Diameter (larger = more suspicious, threshold at 6mm)
    if 'tbp_lv_minorAxisMM' in df.columns:
        features['abcde_diameter'] = df['tbp_lv_minorAxisMM']
        features['abcde_large_lesion'] = (df['tbp_lv_minorAxisMM'] > 6).astype(int)

    # Shape compactness (circle = 1, irregular = higher)
    if 'tbp_lv_areaMM2' in df.columns and 'tbp_lv_perimeterMM' in df.columns:
        features['abcde_compactness'] = (
            4 * np.pi * df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] ** 2 + 1e-8)
        )
        features['abcde_irregularity'] = 1 - features['abcde_compactness']

    # Color complexity (L*a*b* color space variation)
    if all(col in df.columns for col in ['tbp_lv_deltaL', 'tbp_lv_deltaA', 'tbp_lv_deltaB']):
        features['abcde_color_delta'] = np.sqrt(
            df['tbp_lv_deltaL']**2 + df['tbp_lv_deltaA']**2 + df['tbp_lv_deltaB']**2
        )

    return features


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key variables.

    Multiplicative interactions can capture non-linear relationships.
    """
    features = df.copy()

    # Age × lesion characteristics
    if 'age_approx' in df.columns:
        if 'tbp_lv_areaMM2' in df.columns:
            features['age_x_area'] = df['age_approx'] * df['tbp_lv_areaMM2']
        if 'tbp_lv_color_std_mean' in df.columns:
            features['age_x_color_var'] = df['age_approx'] * df['tbp_lv_color_std_mean']

    # Size × irregularity
    if 'tbp_lv_areaMM2' in df.columns and 'tbp_lv_norm_border' in df.columns:
        features['size_x_border'] = df['tbp_lv_areaMM2'] * df['tbp_lv_norm_border']

    # Asymmetry × color variation
    if 'tbp_lv_symm_2axis' in df.columns and 'tbp_lv_color_std_mean' in df.columns:
        features['asymm_x_color'] = (1 - df['tbp_lv_symm_2axis']) * df['tbp_lv_color_std_mean']

    return features


# ============================================================================
# Metrics
# ============================================================================

def partial_auc_above_tpr(y_true: np.ndarray, y_pred: np.ndarray, min_tpr: float = 0.8) -> float:
    """Calculate partial AUC above minimum TPR (competition metric)"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    valid_idx = tpr >= min_tpr
    if not np.any(valid_idx):
        return 0.0

    valid_tpr = tpr[valid_idx]
    valid_fpr = fpr[valid_idx]

    fpr_min, fpr_max = valid_fpr.min(), valid_fpr.max()
    if fpr_max - fpr_min < 1e-7:
        return 0.0

    norm_fpr = (valid_fpr - fpr_min) / (fpr_max - fpr_min)
    pauc = np.trapz(valid_tpr, norm_fpr)

    return pauc


# ============================================================================
# Data Loading
# ============================================================================

def load_embeddings(config: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ViT and ConvNeXt embeddings"""

    print("Loading embeddings...")

    # Load ViT embeddings
    vit_train = np.load(Path(config.vit_embeddings_dir) / 'train_embeddings.npy')
    vit_test = np.load(Path(config.vit_embeddings_dir) / 'test_embeddings.npy')
    print(f"ViT: train {vit_train.shape}, test {vit_test.shape}")

    # Load ConvNeXt embeddings
    convnext_train = np.load(Path(config.convnext_embeddings_dir) / 'train_embeddings.npy')
    convnext_test = np.load(Path(config.convnext_embeddings_dir) / 'test_embeddings.npy')
    print(f"ConvNeXt: train {convnext_train.shape}, test {convnext_test.shape}")

    # Concatenate embeddings
    X_train_embeddings = np.concatenate([vit_train, convnext_train], axis=1)
    X_test_embeddings = np.concatenate([vit_test, convnext_test], axis=1)

    print(f"Combined: train {X_train_embeddings.shape}, test {X_test_embeddings.shape}")

    return X_train_embeddings, X_test_embeddings, vit_train, convnext_train


def create_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Create final feature matrix combining embeddings and tabular features.

    Returns:
        X_train: (N_train, total_features)
        X_test: (N_test, total_features)
        feature_names: List of feature names
    """
    print("\nEngineering tabular features...")

    # Apply feature engineering to both train and test
    train_features = train_df.copy()
    test_features = test_df.copy()

    # Patient-normalized features
    train_features = create_patient_features(train_features)
    test_features = create_patient_features(test_features)

    # ABCDE clinical features
    train_features = create_abcde_features(train_features)
    test_features = create_abcde_features(test_features)

    # Interaction features
    train_features = create_interaction_features(train_features)
    test_features = create_interaction_features(test_features)

    # Select numeric columns (excluding ID and target)
    exclude_cols = ['isic_id', 'patient_id', 'target', 'fold']
    numeric_cols = [
        col for col in train_features.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_features[col])
    ]

    print(f"Engineered {len(numeric_cols)} tabular features")

    # Extract tabular features
    X_train_tabular = train_features[numeric_cols].fillna(0).values
    X_test_tabular = test_features[numeric_cols].fillna(0).values

    # Combine with embeddings
    X_train = np.concatenate([X_train_tabular, train_embeddings], axis=1)
    X_test = np.concatenate([X_test_tabular, test_embeddings], axis=1)

    # Feature names
    embedding_names = [f'embed_{i}' for i in range(train_embeddings.shape[1])]
    feature_names = numeric_cols + embedding_names

    print(f"\nFinal feature matrix:")
    print(f"  Tabular features: {X_train_tabular.shape[1]}")
    print(f"  Image embeddings: {train_embeddings.shape[1]}")
    print(f"  Total features: {X_train.shape[1]}")

    return X_train, X_test, feature_names


# ============================================================================
# Training
# ============================================================================

def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_test: np.ndarray,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train LightGBM ensemble with 5-fold cross-validation.

    Returns:
        oof_preds: Out-of-fold predictions for train set
        test_preds: Averaged predictions for test set
        metrics: Dictionary of validation metrics per fold
    """
    n_samples = len(X)
    oof_preds = np.zeros(n_samples)
    test_preds = np.zeros(len(X_test))

    skf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_metrics = []

    print("\n" + "="*80)
    print("Training LightGBM Ensemble")
    print("="*80)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        print(f"\n--- Fold {fold} ---")
        print(f"Train: {len(train_idx)} samples ({y[train_idx].sum()} positive)")
        print(f"Val:   {len(val_idx)} samples ({y[val_idx].sum()} positive)")

        # Create datasets
        train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
        val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_data)

        # Train model
        model = lgb.train(
            config.lgb_params,
            train_data,
            num_boost_round=config.num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds),
                lgb.log_evaluation(100)
            ]
        )

        # Out-of-fold predictions
        oof_preds[val_idx] = model.predict(X[val_idx])

        # Test predictions (accumulate for averaging)
        test_preds += model.predict(X_test) / config.n_folds

        # Calculate metrics
        val_auc = roc_auc_score(y[val_idx], oof_preds[val_idx])
        val_pauc = partial_auc_above_tpr(y[val_idx], oof_preds[val_idx], min_tpr=0.8)

        fold_metrics.append({
            'fold': fold,
            'auc': val_auc,
            'pauc': val_pauc,
            'best_iteration': model.best_iteration
        })

        print(f"Val AUC:  {val_auc:.4f}")
        print(f"Val pAUC: {val_pauc:.4f}")
        print(f"Best iteration: {model.best_iteration}")

    # Overall metrics
    print("\n" + "="*80)
    print("Cross-Validation Results")
    print("="*80)

    for metrics in fold_metrics:
        print(f"Fold {metrics['fold']}: AUC={metrics['auc']:.4f}, pAUC={metrics['pauc']:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_pauc = partial_auc_above_tpr(y, oof_preds, min_tpr=0.8)

    print(f"\nOverall OOF:")
    print(f"  AUC:  {overall_auc:.4f}")
    print(f"  pAUC: {overall_pauc:.4f}")

    return oof_preds, test_preds, fold_metrics


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main ensemble training function"""
    config = Config()

    print("="*80)
    print("ISIC 2024 - Multi-Modal Ensemble")
    print("="*80)

    # Load metadata
    print("\nLoading metadata...")
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Positive rate: {train_df['target'].mean()*100:.3f}%")

    # Check for fold column
    if 'fold' not in train_df.columns:
        print("\nCreating cross-validation folds...")
        skf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
        train_df['fold'] = -1
        for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['target'], train_df['patient_id'])):
            train_df.loc[val_idx, 'fold'] = fold

    # Load embeddings
    train_embeddings, test_embeddings, vit_train, convnext_train = load_embeddings(config)

    # Create feature matrix
    X_train, X_test, feature_names = create_feature_matrix(
        train_df, test_df, train_embeddings, test_embeddings
    )

    # Extract target and groups
    y_train = train_df['target'].values
    groups = train_df['patient_id'].values

    # Train ensemble
    oof_preds, test_preds, fold_metrics = train_ensemble(
        X_train, y_train, groups, X_test, config
    )

    # Save OOF predictions
    oof_df = pd.DataFrame({
        'isic_id': train_df['isic_id'],
        'target': y_train,
        'prediction': oof_preds
    })
    oof_df.to_csv(Path(config.output_dir) / 'oof_predictions.csv', index=False)
    print(f"\nSaved OOF predictions to: {config.output_dir}/oof_predictions.csv")

    # Create submission
    submission = pd.DataFrame({
        'isic_id': test_df['isic_id'],
        'target': test_preds
    })
    submission.to_csv(Path(config.output_dir) / config.submission_file, index=False)

    print(f"\n{'='*80}")
    print("ENSEMBLE COMPLETE")
    print(f"{'='*80}")
    print(f"\nSubmission saved to: {config.output_dir}/{config.submission_file}")
    print(f"Ready to submit to Kaggle!")


if __name__ == '__main__':
    main()
