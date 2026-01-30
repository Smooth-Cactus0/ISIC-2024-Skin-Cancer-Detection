<p align="center">
  <h1 align="center">Skin Cancer Detection with 3D Total Body Photography</h1>
  <p align="center">
    Multi-modal deep learning pipeline for the ISIC 2024 Kaggle Challenge
    <br />
    <em>Combining Vision Transformers, ConvNeXt, and Gradient Boosting for clinical-grade melanoma screening</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/timm-0.9%2B-green" alt="timm">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Kaggle-ISIC%202024-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
</p>

---

## Executive Summary

This project implements a **multi-modal ensemble pipeline** for binary classification of skin lesions — distinguishing malignant melanoma from benign lesions using 3D Total Body Photography (3D-TBP). The system processes both lesion images and rich clinical metadata to produce predictions evaluated on **partial AUC above 80% True Positive Rate (pAUC@80TPR)**, a metric specifically designed for clinical screening where missing a malignant lesion is far more costly than a false positive.

The approach synthesizes techniques from the **top 5 Kaggle solutions** into a single, well-documented, reproducible pipeline — making it suitable for both competition submission and as a demonstration of medical AI engineering practices.

### Why This Matters

Melanoma is the deadliest form of skin cancer, responsible for **~8,000 deaths annually** in the United States alone. Early detection through automated screening can reduce mortality by catching malignant lesions before they metastasize. The ISIC 2024 Challenge provides the first large-scale dataset of **3D Total Body Photography crops** — a new imaging modality that captures lesions in standardized conditions across 9 international dermatology institutions.

---

## Key Results

| Model | Validation pAUC@80TPR | Description |
|-------|----------------------|-------------|
| LightGBM (tabular baseline) | 0.10–0.12 | GBDT on raw metadata features |
| + Patient-normalized features | 0.12–0.14 | Relative features per patient |
| EVA02 ViT (image only) | 0.14–0.16 | Fine-tuned Vision Transformer |
| ConvNeXt V2 (image only) | 0.13–0.15 | Modern CNN for diversity |
| **Multi-modal ensemble** | **0.16–0.18** | **Images + tabular + stacking** |
| Top 3 Kaggle solutions | 0.17–0.18 | Competition reference |

> **Note**: Results are expected benchmarks based on architecture choices and top solution analysis. See [docs/results.md](docs/results.md) for detailed ablation analysis.

---

## Clinical Context

### The ABCDE of Melanoma Detection

Clinical dermatologists use the **ABCDE criteria** to evaluate suspicious lesions:

| Criterion | Clinical Meaning | Computational Proxy |
|-----------|-----------------|---------------------|
| **A**symmetry | Irregular shape | Compactness ratio, perimeter² / area |
| **B**order | Ragged edges | Border irregularity index |
| **C**olor | Multiple colors | Color heterogeneity, TBP color std |
| **D**iameter | > 6mm | Lesion area from TBP measurements |
| **E**volution | Changing over time | Not available in cross-sectional data |

This pipeline encodes A–D as engineered features, complementing the raw TBP device measurements.

### Why pAUC@80% TPR?

Standard AUC treats all operating points equally. In melanoma screening:
- **False negatives** (missing cancer) → delayed treatment → potential death
- **False positives** (flagging benign) → unnecessary biopsy → minor inconvenience

The competition metric **pAUC above 80% TPR** evaluates the model *only* in the clinically relevant region where at least 80% of malignant lesions are detected, heavily penalizing models that sacrifice sensitivity for specificity.

### Extreme Class Imbalance

The dataset contains **~393 malignant vs ~400,000 benign samples** — a **1,000:1 imbalance ratio**. This requires specialized techniques:
- **Balanced batch sampling**: Each mini-batch contains 50% positive samples
- **Focal loss**: Down-weights easy (benign) examples, focuses learning on ambiguous cases
- **Patient-normalized features**: Relative measurements more discriminative than absolute values

---

## Architecture

### Multi-Modal Ensemble Pipeline

```
                           ISIC 2024 Dataset
                    ┌──────────┴──────────┐
                    │                     │
            ┌───────▼───────┐    ┌───────▼───────┐
            │  3D-TBP Image │    │   Clinical    │
            │  Crops (JPEG) │    │   Metadata    │
            │  128×128 px   │    │   50+ fields  │
            └───────┬───────┘    └───────┬───────┘
                    │                     │
         ┌──────────┴──────────┐          │
         │                     │          │
   ┌─────▼─────┐        ┌─────▼─────┐    │
   │   EVA02   │        │ ConvNeXt  │    │
   │  ViT-Base │        │  V2 Nano  │    │
   │  (768-d)  │        │  (640-d)  │    │
   └─────┬─────┘        └─────┬─────┘    │
         │                     │          │
         └──────────┬──────────┘          │
                    │                     │
            ┌───────▼───────┐    ┌───────▼───────┐
            │    Image      │    │   Feature     │
            │  Embeddings   │    │  Engineering  │
            │  (768+640-d)  │    │  Pipeline     │
            └───────┬───────┘    └───────┬───────┘
                    │                     │
                    │    ┌────────────────┤
                    │    │                │
                    │    │    ┌───────────▼──────────┐
                    │    │    │  Patient-Normalized  │
                    │    │    │  • Z-score per patient│
                    │    │    │  • Percentile rank    │
                    │    │    │  • ABCDE criteria     │
                    │    │    └───────────┬──────────┘
                    │    │                │
                    └────┴────────┬───────┘
                                 │
                    ┌────────────▼────────────┐
                    │    GBDT Ensemble        │
                    │  ┌────────┬────────┐    │
                    │  │LightGBM│XGBoost │    │
                    │  │        │CatBoost│    │
                    │  └────┬───┴───┬────┘    │
                    │       │  Rank │         │
                    │       │Average│         │
                    └───────┴───┬───┴─────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Final Prediction   │
                    │  P(malignant)       │
                    │  Evaluated on       │
                    │  pAUC@80% TPR       │
                    └─────────────────────┘
```

### Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Image backbone** | EVA02 ViT-Base | Best performance/speed in top solutions; ImageNet-21K pretraining captures medical textures |
| **Second backbone** | ConvNeXtV2 Nano | Captures local patterns (borders, texture) complementary to ViT's global attention |
| **Loss function** | Focal Loss (α=0.25, γ=2.0) | Explicitly addresses 1000:1 imbalance; superior to class weights alone |
| **CV strategy** | 5-fold StratifiedGroupKFold | Patient-level grouping prevents data leakage from multi-lesion patients |
| **Final model** | GBDT stacking | Gradient boosting on embeddings + tabular features; robust to feature noise |
| **Feature engineering** | Patient normalization | Relative features (z-score, percentile) more discriminative than absolute values |
| **Training** | Mixed precision (FP16) | 2× faster training, critical for Kaggle GPU time limits |

---

## Project Structure

```
├── notebooks/                  # Interactive analysis
│   ├── 01_eda.ipynb           # Exploratory data analysis: class distribution,
│   │                          #   demographics, TBP features, patient patterns
│   └── 02_baseline_tabular.ipynb  # GBDT baseline with feature engineering,
│                              #   SHAP analysis, OOF predictions
│
├── src/                       # Core library modules
│   ├── data/
│   │   ├── dataset.py         # ISICDataset, BalancedBatchSampler, fold splits
│   │   └── augmentations.py   # Albumentations pipelines, MixUp, TTA, hair removal
│   ├── features/
│   │   └── patient_features.py # Patient normalization, ABCDE criteria, aggregates
│   ├── models/
│   │   ├── vit_classifier.py  # EVA02/ViT classifiers, FocalLoss, ensemble wrapper
│   │   └── convnext_classifier.py # ConvNeXtV2 classifier, HybridModel fusion
│   └── utils/
│       └── metrics.py         # pAUC@80TPR (competition metric), clinical metrics
│
├── scripts/                   # Training and inference pipelines
│   ├── train_vit.py           # Full ViT training: CV, early stopping, checkpoints
│   ├── train_convnext.py      # ConvNeXt training pipeline
│   └── extract_embeddings.py  # OOF embedding extraction for GBDT stacking
│
├── kaggle_notebooks/          # Self-contained Kaggle GPU scripts
│   ├── train_vit_kaggle.py    # EVA02 training (~10h on P100)
│   ├── train_convnext_kaggle.py   # ConvNeXt training (~8h)
│   ├── extract_embeddings_kaggle.py # Embedding extraction (~1h per model)
│   ├── ensemble_kaggle.py     # Final GBDT ensemble + submission
│   └── README.md              # Step-by-step deployment guide
│
├── configs/                   # YAML experiment configurations
│   ├── eva02.yaml             # EVA02 small config
│   ├── eva02_production.yaml  # EVA02 base production settings
│   ├── vit_base.yaml          # ViT-Base config
│   └── convnext_base.yaml     # ConvNeXt-Base config
│
├── docs/                      # Technical documentation
│   ├── methodology.md         # Problem formulation, approach, CV design
│   ├── model_discussion.md    # Architecture deep dive, loss functions
│   ├── results.md             # Benchmarks, ablations, error analysis
│   └── kaggle_deployment.md   # Kaggle GPU training guide
│
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT License
```

---

## Getting Started

### Prerequisites

- **Python 3.10 or 3.11** (not 3.14 — compatibility issues with PyTorch/timm)
- **CUDA-capable GPU** with ≥ 8GB VRAM (for image model training)
- **Kaggle account** with [API credentials](https://www.kaggle.com/docs/api) configured

### Installation

```bash
# Clone the repository
git clone https://github.com/Smooth-Cactus0/ISIC-2024-Skin-Cancer-Detection.git
cd ISIC-2024-Skin-Cancer-Detection

# Create conda environment
conda create -n isic python=3.11
conda activate isic

# Install dependencies
pip install -r requirements.txt

# Download competition data
kaggle competitions download -c isic-2024-challenge -p data/raw/
unzip data/raw/isic-2024-challenge.zip -d data/raw/
```

### Quick Start — Local Training

```bash
# 1. Train EVA02 Vision Transformer (single fold for testing)
python scripts/train_vit.py --use_sample --fold 0 --epochs 5

# 2. Train ConvNeXt (single fold)
python scripts/train_convnext.py --use_sample --fold 0 --epochs 5

# 3. Extract embeddings for GBDT stacking
python scripts/extract_embeddings.py --model outputs/eva02_best.pth

# 4. Full 5-fold training with production config
python scripts/train_vit.py --config configs/eva02_production.yaml
```

### Quick Start — Kaggle GPU Training

All code is packaged as **self-contained scripts** for Kaggle's free GPU environment. See the [Kaggle deployment guide](kaggle_notebooks/README.md) for the complete workflow.

```
Session 1 (~10h): train_vit_kaggle.py        → EVA02 training
Session 2 (~8h):  train_convnext_kaggle.py    → ConvNeXt training
Session 3 (~1h):  extract_embeddings_kaggle.py → Embedding extraction
Session 4 (~1h):  ensemble_kaggle.py          → Final ensemble + submission
```

---

## Notebooks Guide

### 01 — Exploratory Data Analysis ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb))

Comprehensive analysis of the ISIC 2024 SLICE-3D dataset:
- **Class distribution**: Visualizing the extreme 1000:1 imbalance
- **Patient-level analysis**: Lesion count per patient, malignancy clustering
- **Demographics**: Age, sex, and anatomic site distributions by class
- **TBP feature analysis**: 50+ device-measured features, correlation structure
- **Sample images**: Comparing malignant vs. benign crops
- **Cross-validation design**: Demonstrating patient-level split necessity

### 02 — Baseline Tabular Model ([`notebooks/02_baseline_tabular.ipynb`](notebooks/02_baseline_tabular.ipynb))

Feature engineering and gradient boosting baseline:
- **Raw metadata baseline**: LightGBM on TBP features
- **Patient-normalized features**: Z-score, percentile rank, deviation
- **ABCDE clinical features**: Computational proxies for dermatology criteria
- **Feature importance**: SHAP analysis of top predictive features
- **Out-of-fold predictions**: Saved for downstream ensemble

---

## Technical Documentation

| Document | Description |
|----------|-------------|
| [Methodology](docs/methodology.md) | Problem formulation, evaluation metric rationale, feature engineering approach, cross-validation design |
| [Model Discussion](docs/model_discussion.md) | Deep dive into EVA02, ConvNeXt, Focal Loss, patient normalization, ensemble diversity theory |
| [Results Analysis](docs/results.md) | Expected benchmarks, ablation tables, feature importance, error analysis, comparison to top Kaggle solutions |
| [Kaggle Deployment](docs/kaggle_deployment.md) | Step-by-step guide for training on Kaggle's free GPU, session management, troubleshooting |

---

## References

### Papers & Datasets
- Cassidy, B. et al. (2024). [Analysis of the ISIC 2024 SLICE-3D Dataset](https://www.nature.com/articles/s41597-024-03743-w). *Scientific Data*.
- Fang, Y. et al. (2023). [EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/abs/2303.11331). *arXiv*.
- Woo, S. et al. (2023). [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808). *CVPR*.
- Lin, T.-Y. et al. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). *ICCV*.

### Competition & Solutions
- [ISIC 2024 Challenge — Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)
- [1st Place Solution — Multi-modal EVA02 Ensemble](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532538)
- [3rd Place Solution — ViT/ConvNeXt Ensemble (Code)](https://github.com/kyohei-123/kaggle-isic-2024-3rd-place-solution)

---

## Author

**Alexy Louis**

Built as a demonstration of medical AI engineering — combining clinical domain knowledge, state-of-the-art deep learning, and rigorous evaluation methodology.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
