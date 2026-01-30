"""
Evaluation metrics for ISIC 2024 Skin Cancer Detection.

Primary metric: Partial AUC (pAUC) above 80% True Positive Rate
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


def partial_auc_above_tpr(y_true, y_pred, min_tpr=0.8):
    """
    Calculate partial AUC above a minimum TPR threshold.

    This is the competition metric for ISIC 2024. It focuses on the high-sensitivity
    region of the ROC curve, which is critical for medical screening applications
    where missing a malignant lesion has severe consequences.

    Args:
        y_true: Ground truth binary labels (0=benign, 1=malignant)
        y_pred: Predicted probabilities or scores
        min_tpr: Minimum true positive rate threshold (default: 0.8)

    Returns:
        Partial AUC score normalized to [0, 1] range

    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        >>> pauc = partial_auc_above_tpr(y_true, y_pred, min_tpr=0.8)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find region where TPR >= min_tpr
    valid_idx = tpr >= min_tpr

    if not np.any(valid_idx):
        # No region meets the TPR threshold
        return 0.0

    # Extract valid region
    valid_tpr = tpr[valid_idx]
    valid_fpr = fpr[valid_idx]

    # Calculate partial AUC using trapezoidal integration
    # Note: We integrate over FPR, so we need to sort by FPR
    sort_idx = np.argsort(valid_fpr)
    valid_fpr = valid_fpr[sort_idx]
    valid_tpr = valid_tpr[sort_idx]

    # Calculate area under curve in the valid region
    pauc = auc(valid_fpr, valid_tpr)

    # Normalize by the maximum possible area in this region
    # Max area = (max_fpr - min_fpr) * (max_tpr - min_tpr)
    max_fpr = valid_fpr[-1]
    min_fpr = valid_fpr[0]
    max_tpr = valid_tpr[-1]

    max_possible_area = (max_fpr - min_fpr) * (max_tpr - min_tpr)

    if max_possible_area > 0:
        normalized_pauc = pauc / max_possible_area
    else:
        normalized_pauc = 0.0

    return normalized_pauc


def competition_metric(y_true, y_pred):
    """
    Official ISIC 2024 competition metric: pAUC above 80% TPR.

    Wrapper around partial_auc_above_tpr for convenience.
    """
    return partial_auc_above_tpr(y_true, y_pred, min_tpr=0.8)


def sensitivity_at_specificity(y_true, y_pred, target_specificity=0.8):
    """
    Calculate sensitivity (recall) at a target specificity.

    Useful for medical applications where we want to maintain a minimum
    specificity (to avoid too many false positives) while maximizing
    sensitivity.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        target_specificity: Desired specificity level (default: 0.8)

    Returns:
        Sensitivity value at the target specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specificity = 1 - fpr

    # Find threshold closest to target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))

    return tpr[idx]


def specificity_at_sensitivity(y_true, y_pred, target_sensitivity=0.8):
    """
    Calculate specificity at a target sensitivity (TPR).

    For ISIC 2024, we want to know: at 80% sensitivity (catching 80% of
    malignancies), what's our specificity (avoiding false positives)?

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        target_sensitivity: Desired sensitivity level (default: 0.8)

    Returns:
        Specificity value at the target sensitivity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))

    specificity = 1 - fpr[idx]
    return specificity


def optimal_threshold_for_tpr(y_true, y_pred, target_tpr=0.8):
    """
    Find the decision threshold that achieves a target TPR.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        target_tpr: Target true positive rate (default: 0.8)

    Returns:
        Threshold value that achieves the target TPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find threshold closest to target TPR
    idx = np.argmin(np.abs(tpr - target_tpr))

    return thresholds[idx]


def evaluate_binary_classification(y_true, y_pred, threshold=0.5, verbose=True):
    """
    Comprehensive evaluation of binary classification performance.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold (default: 0.5)
        verbose: Print results (default: True)

    Returns:
        Dictionary containing all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, average_precision_score
    )

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_true, y_pred),
        'pauc_80tpr': partial_auc_above_tpr(y_true, y_pred, min_tpr=0.8),
        'average_precision': average_precision_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'sensitivity_at_80spec': sensitivity_at_specificity(y_true, y_pred, 0.8),
        'specificity_at_80sens': specificity_at_sensitivity(y_true, y_pred, 0.8),
        'threshold': threshold
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['confusion_matrix'] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    if verbose:
        print("=" * 80)
        print("BINARY CLASSIFICATION EVALUATION")
        print("=" * 80)
        print(f"\nROC Metrics:")
        print(f"  AUC (full ROC):            {metrics['auc']:.4f}")
        print(f"  pAUC @ 80% TPR (ISIC):     {metrics['pauc_80tpr']:.4f}  ⭐ COMPETITION METRIC")
        print(f"  Average Precision (PR):    {metrics['average_precision']:.4f}")

        print(f"\nClassification Metrics (threshold={threshold:.2f}):")
        print(f"  Accuracy:                  {metrics['accuracy']:.4f}")
        print(f"  Precision:                 {metrics['precision']:.4f}")
        print(f"  Recall (Sensitivity):      {metrics['recall']:.4f}")
        print(f"  F1 Score:                  {metrics['f1']:.4f}")

        print(f"\nClinical Metrics:")
        print(f"  Sensitivity @ 80% Spec:    {metrics['sensitivity_at_80spec']:.4f}")
        print(f"  Specificity @ 80% Sens:    {metrics['specificity_at_80sens']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Neg    Pos")
        print(f"  Actual  Neg   {tn:6d} {fp:6d}")
        print(f"          Pos   {fn:6d} {tp:6d}")
        print(f"\n  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}  ⚠️  CRITICAL FOR MEDICAL SCREENING")
        print(f"  True Positives:  {tp:,}")

    return metrics


if __name__ == "__main__":
    # Test the metrics
    print("Testing metrics implementation...\n")

    # Simulate predictions
    np.random.seed(42)
    n_samples = 1000
    n_positive = 50  # Imbalanced like ISIC

    y_true = np.zeros(n_samples)
    y_true[:n_positive] = 1

    # Good model: higher scores for positives
    y_pred = np.random.beta(2, 5, n_samples)
    y_pred[:n_positive] += 0.3  # Boost positive class scores
    y_pred = np.clip(y_pred, 0, 1)

    # Evaluate
    metrics = evaluate_binary_classification(y_true, y_pred, threshold=0.5)

    print(f"\n{'=' * 80}")
    print("✓ Metrics module tested successfully")
