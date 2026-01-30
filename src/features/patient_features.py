"""
Patient-normalized feature engineering for ISIC 2024.

Critical insight from top solutions: Features normalized within patient context
are highly discriminative. A "large" lesion for one patient may be "small" for another.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class PatientFeatureTransformer:
    """
    Transform features to be relative to patient-specific distributions.

    This addresses the patient-level correlation in the data by creating features
    that capture how abnormal a lesion is relative to the patient's other lesions.

    Example:
        >>> transformer = PatientFeatureTransformer(features=['tbp_lv_areaMM2', 'tbp_lv_color_std_mean'])
        >>> df_train = transformer.fit_transform(df_train, patient_col='patient_id')
        >>> df_test = transformer.transform(df_test, patient_col='patient_id')
    """

    def __init__(
        self,
        features: List[str],
        operations: List[str] = ['z_score', 'percentile', 'deviation'],
        min_lesions_for_normalization: int = 2
    ):
        """
        Initialize patient feature transformer.

        Args:
            features: List of feature column names to normalize
            operations: List of operations to apply:
                - 'z_score': (value - patient_mean) / patient_std
                - 'percentile': Rank of value within patient's lesions
                - 'deviation': Absolute deviation from patient mean
                - 'is_max': Binary flag if this is patient's maximum value
                - 'is_min': Binary flag if this is patient's minimum value
            min_lesions_for_normalization: Minimum lesions required for patient-specific stats
                (uses population stats for patients with fewer lesions)
        """
        self.features = features
        self.operations = operations
        self.min_lesions = min_lesions_for_normalization

        # Will store population statistics for fallback
        self.population_mean_ = {}
        self.population_std_ = {}

    def fit(self, df: pd.DataFrame, patient_col: str = 'patient_id'):
        """
        Fit on training data to learn population statistics.

        Args:
            df: Training dataframe
            patient_col: Column name for patient identifier

        Returns:
            self
        """
        for feat in self.features:
            if feat in df.columns:
                self.population_mean_[feat] = df[feat].mean()
                self.population_std_[feat] = df[feat].std()

        return self

    def transform(self, df: pd.DataFrame, patient_col: str = 'patient_id') -> pd.DataFrame:
        """
        Transform features to patient-normalized versions.

        Args:
            df: Dataframe to transform
            patient_col: Column name for patient identifier

        Returns:
            Dataframe with additional patient-normalized features
        """
        df = df.copy()

        for feat in self.features:
            if feat not in df.columns:
                continue

            # Calculate patient-level statistics
            patient_stats = df.groupby(patient_col)[feat].agg(['mean', 'std', 'count', 'min', 'max'])
            patient_stats.columns = [f'{feat}_patient_mean', f'{feat}_patient_std',
                                    f'{feat}_patient_count', f'{feat}_patient_min', f'{feat}_patient_max']

            # Merge back to original dataframe
            df = df.merge(patient_stats, left_on=patient_col, right_index=True, how='left')

            # Apply requested operations
            if 'z_score' in self.operations:
                # Z-score within patient
                df[f'{feat}_patient_zscore'] = self._safe_zscore(
                    df[feat],
                    df[f'{feat}_patient_mean'],
                    df[f'{feat}_patient_std'],
                    df[f'{feat}_patient_count'],
                    self.population_mean_.get(feat, 0),
                    self.population_std_.get(feat, 1)
                )

            if 'percentile' in self.operations:
                # Percentile rank within patient's lesions
                df[f'{feat}_patient_percentile'] = df.groupby(patient_col)[feat].rank(pct=True)

            if 'deviation' in self.operations:
                # Absolute deviation from patient mean
                df[f'{feat}_patient_deviation'] = np.abs(df[feat] - df[f'{feat}_patient_mean'])

            if 'is_max' in self.operations:
                # Is this the patient's largest value?
                df[f'{feat}_is_patient_max'] = (df[feat] == df[f'{feat}_patient_max']).astype(int)

            if 'is_min' in self.operations:
                # Is this the patient's smallest value?
                df[f'{feat}_is_patient_min'] = (df[feat] == df[f'{feat}_patient_min']).astype(int)

            # Clean up intermediate columns
            df = df.drop(columns=[
                f'{feat}_patient_mean', f'{feat}_patient_std', f'{feat}_patient_count',
                f'{feat}_patient_min', f'{feat}_patient_max'
            ])

        return df

    def fit_transform(self, df: pd.DataFrame, patient_col: str = 'patient_id') -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, patient_col).transform(df, patient_col)

    def _safe_zscore(self, values, patient_mean, patient_std, patient_count,
                     pop_mean, pop_std):
        """
        Calculate z-score with fallback to population statistics.

        For patients with insufficient lesions, use population statistics.
        """
        # Use patient stats if enough lesions, otherwise use population
        use_patient_stats = patient_count >= self.min_lesions

        mean = np.where(use_patient_stats, patient_mean, pop_mean)
        std = np.where(use_patient_stats, patient_std, pop_std)

        # Avoid division by zero
        std = np.where(std < 1e-6, 1.0, std)

        zscore = (values - mean) / std
        return zscore


class ABCDEFeatureEngineer:
    """
    Create features inspired by clinical ABCDE criteria for melanoma.

    A - Asymmetry
    B - Border irregularity
    C - Color variation
    D - Diameter
    E - Evolution (not available in static dataset)
    """

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ABCDE-inspired features.

        Args:
            df: Input dataframe with TBP features

        Returns:
            Dataframe with additional engineered features
        """
        df = df.copy()

        # A - Asymmetry features
        if 'tbp_lv_symm_2axis' in df.columns:
            # Lower symmetry = higher asymmetry (more suspicious)
            df['asymmetry_score'] = 1 - df['tbp_lv_symm_2axis']

        if 'tbp_lv_eccentricity' in df.columns:
            # Eccentricity indicates elongation (asymmetry)
            df['shape_asymmetry'] = df['tbp_lv_eccentricity']

        # B - Border irregularity features
        if 'tbp_lv_norm_border' in df.columns:
            df['border_irregularity'] = df['tbp_lv_norm_border']

        if 'tbp_lv_areaMM2' in df.columns and 'tbp_lv_perimeterMM' in df.columns:
            # Compactness: circular lesions have high compactness
            # Irregular borders have low compactness
            df['compactness'] = (4 * np.pi * df['tbp_lv_areaMM2']) / (df['tbp_lv_perimeterMM']**2 + 1e-6)
            df['border_complexity'] = df['tbp_lv_perimeterMM'] / (np.sqrt(df['tbp_lv_areaMM2']) + 1e-6)

        # C - Color variation features
        if 'tbp_lv_color_std_mean' in df.columns:
            df['color_heterogeneity'] = df['tbp_lv_color_std_mean']

        # Color deltas capture variation
        delta_cols = [col for col in df.columns if 'tbp_lv_delta' in col.lower()]
        if delta_cols:
            # Combined color variation score
            df['color_variation_score'] = df[delta_cols].abs().mean(axis=1)

        # RGB-like ratios (from LAB color space)
        if 'tbp_lv_A' in df.columns and 'tbp_lv_B' in df.columns:
            df['color_ratio_AB'] = df['tbp_lv_A'] / (np.abs(df['tbp_lv_B']) + 1e-6)

        if 'tbp_lv_L' in df.columns and 'tbp_lv_Lext' in df.columns:
            # Lightness contrast between center and exterior
            df['lightness_contrast'] = np.abs(df['tbp_lv_L'] - df['tbp_lv_Lext'])

        # D - Diameter features
        if 'tbp_lv_areaMM2' in df.columns:
            # Equivalent diameter from area
            df['equivalent_diameter'] = 2 * np.sqrt(df['tbp_lv_areaMM2'] / np.pi)

        if 'clin_size_long_diam_mm' in df.columns:
            df['size_mm'] = df['clin_size_long_diam_mm']

        # Interaction features (combining ABCDE components)
        if 'asymmetry_score' in df.columns and 'color_heterogeneity' in df.columns:
            df['asymmetry_x_color'] = df['asymmetry_score'] * df['color_heterogeneity']

        if 'border_complexity' in df.columns and 'color_variation_score' in df.columns:
            df['border_x_color'] = df['border_complexity'] * df['color_variation_score']

        return df


class AggregatePatientFeatures:
    """
    Create patient-level aggregate features.

    These capture patient characteristics like total lesion count,
    average lesion size, etc.
    """

    @staticmethod
    def create_features(df: pd.DataFrame, patient_col: str = 'patient_id') -> pd.DataFrame:
        """
        Create patient-level aggregate features.

        Args:
            df: Input dataframe
            patient_col: Column name for patient identifier

        Returns:
            Dataframe with additional patient aggregate features
        """
        df = df.copy()

        # Count lesions per patient
        patient_lesion_count = df.groupby(patient_col).size()
        df['patient_lesion_count'] = df[patient_col].map(patient_lesion_count)

        # Patient-level statistics for key features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        tbp_cols = [col for col in numeric_cols if col.startswith('tbp_lv_')]

        if tbp_cols:
            # Average TBP features across patient's lesions
            patient_means = df.groupby(patient_col)[tbp_cols].transform('mean')
            patient_stds = df.groupby(patient_col)[tbp_cols].transform('std')

            # Add a few key aggregate features (not all to avoid feature explosion)
            if 'tbp_lv_areaMM2' in df.columns:
                df['patient_avg_lesion_size'] = patient_means['tbp_lv_areaMM2']
                df['patient_std_lesion_size'] = patient_stds['tbp_lv_areaMM2']

            if 'tbp_lv_color_std_mean' in df.columns:
                df['patient_avg_color_variation'] = patient_means['tbp_lv_color_std_mean']

        return df


def create_all_features(
    df: pd.DataFrame,
    patient_col: str = 'patient_id',
    include_patient_normalized: bool = True,
    include_abcde: bool = True,
    include_aggregates: bool = True
) -> pd.DataFrame:
    """
    Convenience function to create all engineered features.

    Args:
        df: Input dataframe
        patient_col: Column name for patient identifier
        include_patient_normalized: Include patient-normalized features
        include_abcde: Include ABCDE clinical features
        include_aggregates: Include patient aggregate features

    Returns:
        Dataframe with all engineered features
    """
    df = df.copy()

    # ABCDE features first (base features)
    if include_abcde:
        df = ABCDEFeatureEngineer.create_features(df)

    # Patient aggregates
    if include_aggregates:
        df = AggregatePatientFeatures.create_features(df, patient_col)

    # Patient-normalized features
    if include_patient_normalized:
        # Select numeric TBP features to normalize
        tbp_cols = [col for col in df.columns if col.startswith('tbp_lv_')]
        numeric_tbp = df[tbp_cols].select_dtypes(include=[np.number]).columns.tolist()

        if numeric_tbp:
            transformer = PatientFeatureTransformer(
                features=numeric_tbp[:10],  # Limit to avoid feature explosion
                operations=['z_score', 'percentile']
            )
            df = transformer.fit_transform(df, patient_col)

    return df


if __name__ == "__main__":
    # Test feature engineering
    print("Testing patient feature engineering...\n")

    # Create sample data
    np.random.seed(42)
    n_patients = 10
    lesions_per_patient = np.random.randint(1, 20, n_patients)

    data = []
    for patient_id in range(n_patients):
        for _ in range(lesions_per_patient[patient_id]):
            data.append({
                'patient_id': f'P{patient_id:03d}',
                'tbp_lv_areaMM2': np.random.gamma(2, 2),
                'tbp_lv_color_std_mean': np.random.beta(2, 5),
                'tbp_lv_symm_2axis': np.random.uniform(0.5, 1.0),
                'tbp_lv_norm_border': np.random.uniform(0, 5)
            })

    df = pd.DataFrame(data)
    print(f"Created sample data: {len(df)} lesions from {n_patients} patients\n")

    # Test patient normalization
    transformer = PatientFeatureTransformer(
        features=['tbp_lv_areaMM2', 'tbp_lv_color_std_mean'],
        operations=['z_score', 'percentile']
    )
    df_transformed = transformer.fit_transform(df)

    print("Patient-normalized features created:")
    print(df_transformed.filter(regex='patient').columns.tolist())
    print(f"\n{df_transformed[['patient_id', 'tbp_lv_areaMM2', 'tbp_lv_areaMM2_patient_zscore', 'tbp_lv_areaMM2_patient_percentile']].head()}")

    # Test ABCDE features
    df_abcde = ABCDEFeatureEngineer.create_features(df)
    print(f"\n\nABCDE features created:")
    abcde_feats = [col for col in df_abcde.columns if col not in df.columns]
    print(abcde_feats)

    print("\n✓ Feature engineering modules tested successfully")
