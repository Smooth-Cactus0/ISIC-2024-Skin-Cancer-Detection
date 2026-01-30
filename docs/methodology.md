# Methodology

## Problem Formulation

The ISIC 2024 Challenge tasks participants with **binary classification of skin lesions** from 3D Total Body Photography (3D-TBP) crops. Each sample consists of a 15mm × 15mm lesion crop (~128×128 pixels) paired with 50+ clinical metadata fields from the TBP Lesion Visualizer device.

### Input Space
- **Images**: JPEG crops from 3D-TBP, standardized field-of-view
- **Metadata**: Patient demographics (age, sex, anatomic site) + TBP device measurements (color statistics, shape indices, symmetry metrics, border complexity)
- **Source**: 9 international dermatology institutions (multi-site, multi-device)

### Output
- **P(malignant)**: Continuous probability score per lesion
- **Evaluation**: Partial AUC above 80% True Positive Rate (pAUC@80TPR)

### Why pAUC@80TPR?

Standard ROC-AUC treats all operating points equally, but clinical melanoma screening demands **high sensitivity**:

```
Full AUC:     Evaluates 0% ≤ TPR ≤ 100%  →  gives equal weight to all thresholds
pAUC@80TPR:   Evaluates 80% ≤ TPR ≤ 100% →  only rewards models that detect ≥80% of cancers
```

This metric reflects the clinical reality: a screening tool that misses 30% of malignant lesions is clinically useless, regardless of how few false positives it generates. The partial AUC constraint forces models to optimize for the high-sensitivity regime.

**Implementation**: `sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=0.2)` computes the area under the ROC curve in the region where FPR ≤ 0.2 (equivalent to TPR ≥ 0.8 for well-calibrated models). Our custom implementation in `src/utils/metrics.py` adds additional clinical metrics.

---

## Dataset Characteristics

### Extreme Class Imbalance

| Class | Count | Proportion |
|-------|-------|------------|
| Benign (0) | ~400,000 | 99.9% |
| Malignant (1) | ~393 | 0.1% |
| **Ratio** | **~1,000:1** | |

This is one of the most extreme imbalance ratios in medical imaging competitions. Standard training would result in a model that predicts "benign" for everything (99.9% accuracy, 0% sensitivity).

### Multi-Site Data Collection

The dataset spans 9 institutions with different:
- Patient demographics (age distributions, skin types)
- Device calibration settings
- Image quality characteristics

This makes the task a **domain generalization** problem — models must learn features that transfer across clinical sites.

### Patient-Level Structure

Individual patients contribute multiple lesions (median ~20 lesions per patient). This creates a critical **data leakage risk**: if train/validation splits are made at the lesion level, the model may learn patient-specific features (skin tone, device settings) rather than lesion-level malignancy signals.

---

## Feature Engineering Approach

### 1. Raw TBP Features

The TBP Lesion Visualizer device outputs 50+ measurements per lesion:
- **Color**: Mean RGB, LAB, HSV values for lesion and surrounding skin
- **Shape**: Area, perimeter, compactness, eccentricity
- **Contrast**: Color difference between lesion and surrounding skin
- **Symmetry**: Multi-axis symmetry scores
- **Border**: Border irregularity, sharpness metrics

### 2. Patient-Normalized Features

**Key insight from top solutions**: Absolute TBP values are confounded by patient-level factors (skin tone, device settings). Normalizing features *relative to a patient's other lesions* isolates the lesion-specific signal.

For each numeric TBP feature `f` and patient `p`:
```
z_score(f, p)     = (f - mean(f|patient=p)) / std(f|patient=p)
percentile(f, p)  = rank(f among patient p's lesions) / n_lesions(p)
deviation(f, p)   = f - median(f|patient=p)
is_max(f, p)      = 1 if f == max(f|patient=p) else 0
is_min(f, p)      = 1 if f == min(f|patient=p) else 0
```

This transformation answers: *"Is this lesion unusual compared to the patient's other lesions?"* — the same clinical question a dermatologist asks during total body examination.

### 3. ABCDE Clinical Features

Computational proxies for the clinical ABCDE melanoma criteria:
- **Asymmetry**: `compactness = 4π × area / perimeter²` (perfect circle = 1.0)
- **Border**: `border_irregularity = perimeter / (2 × √(π × area))` (smooth = 1.0)
- **Color**: `color_heterogeneity = std(H, S, V)` across lesion pixels
- **Diameter**: `relative_size = area / median_area(patient)` (large = suspicious)

### 4. Interaction Features

- **Color contrast**: Lesion color – surrounding skin color (high contrast = suspicious)
- **Lightness ratio**: Lesion lightness / surrounding skin lightness
- **Size × color**: Interaction between lesion size and color abnormality

---

## Model Architecture Choices

### Image Models: EVA02 + ConvNeXt

**Why two architectures?**

Vision Transformers and CNNs capture complementary visual features:

| Feature Type | ViT (EVA02) | CNN (ConvNeXt) |
|-------------|-------------|----------------|
| **Receptive field** | Global (full image attention) | Local → global (hierarchical) |
| **Strength** | Global patterns, symmetry | Textures, borders, fine details |
| **Melanoma signal** | Overall lesion structure | Border irregularity, color gradients |

Ensemble diversity from architecturally different models consistently outperforms ensembles of similar architectures.

**EVA02 ViT-Base** was specifically chosen because:
1. Top-performing backbone in 1st and 3rd place Kaggle solutions
2. ImageNet-21K pretraining captures rich texture representations
3. Excellent performance/speed tradeoff for Kaggle time limits

**ConvNeXtV2 Nano** was chosen because:
1. Used by 3rd place solution for ensemble diversity
2. Small model (640-d embeddings) minimizes overfitting on 393 positive samples
3. Masked autoencoder pretraining (ConvNeXt V2) improves feature quality

### Gradient Boosting Stacking

The final prediction layer uses **GBDT models** (LightGBM, XGBoost, CatBoost) rather than a neural network head because:
1. GBDTs handle heterogeneous features (embeddings + tabular) naturally
2. Less prone to overfitting with only 393 positive samples
3. Built-in feature importance for interpretability
4. Fast inference for Kaggle time limits

---

## Cross-Validation Design

### StratifiedGroupKFold

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in sgkf.split(X, y=targets, groups=patient_ids):
    # All lesions from the same patient are in the same fold
    # Class distribution is maintained across folds
    ...
```

**Why GroupKFold?**

Without patient-level grouping, the model could achieve inflated validation scores by memorizing patient-level features (skin tone, device settings) that correlate with lesion labels. GroupKFold ensures the model is evaluated on *unseen patients*, simulating real-world deployment.

**Why Stratified?**

With only 393 positive samples across 5 folds, random splits could leave some folds with as few as 60–90 positives. Stratification ensures each fold has approximately equal positive sample counts (~78 per fold), producing stable pAUC estimates.

### Out-of-Fold Predictions

For the GBDT stacking layer, we use **out-of-fold (OOF) predictions**:
1. Train image model on folds 1–4 → predict fold 5
2. Train image model on folds 1–3, 5 → predict fold 4
3. ... repeat for all folds
4. Concatenate OOF predictions → every sample has a prediction from a model that never saw it
5. Train GBDT on these OOF embeddings + tabular features

This prevents the GBDT from learning to exploit information leakage through the image model's memorized training samples.

---

## Training Pipeline

### Image Model Training

1. **Data loading**: ISICDataset class supports JPEG files or HDF5 for fast I/O
2. **Balanced sampling**: BalancedBatchSampler ensures 50% positive rate per batch
3. **Augmentation**: Three levels (light/medium/heavy) with Albumentations
4. **Optimization**: AdamW with differential learning rates (backbone: 1e-5, head: 1e-4)
5. **Scheduling**: Cosine annealing with warm restarts
6. **Mixed precision**: FP16 via PyTorch's autocast for 2× speedup
7. **Early stopping**: Patience=5 on validation pAUC@80TPR
8. **Checkpointing**: Save best model per fold based on pAUC

### GBDT Training

1. **Feature assembly**: Image embeddings (768+640-d) + engineered tabular features (~100-d)
2. **Hyperparameter tuning**: Optuna with pAUC@80TPR as objective
3. **Ensemble**: Rank-average LightGBM, XGBoost, CatBoost predictions
4. **Calibration**: Isotonic regression for probability calibration (optional)

---

## Data Leakage Prevention Checklist

- [x] Patient-level CV splits (StratifiedGroupKFold with patient_id)
- [x] OOF embedding extraction (no train-on-train leakage)
- [x] Separate augmentation for train vs. validation
- [x] No target leakage in feature engineering (patient stats computed per fold)
- [x] Test set features computed independently of training data
