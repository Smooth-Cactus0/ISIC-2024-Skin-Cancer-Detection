# Source Modules (`src/`)

Core library implementing the skin cancer detection pipeline.

```
src/
├── data/
│   ├── dataset.py          # ISICDataset (JPEG/HDF5), BalancedBatchSampler,
│   │                       # StratifiedGroupKFold split creation
│   └── augmentations.py    # Albumentations pipelines (light/medium/heavy),
│                           # MixUp, Test-Time Augmentation, hair removal
├── features/
│   └── patient_features.py # PatientFeatureTransformer (z-score, percentile),
│                           # ABCDEFeatureEngineer, AggregatePatientFeatures
├── models/
│   ├── vit_classifier.py   # ViTClassifier (EVA02/ViT), FocalLoss,
│   │                       # EnsembleViT, MODEL_CONFIGS
│   └── convnext_classifier.py # ConvNeXtClassifier, HybridModel (ViT+CNN fusion),
│                              # CONVNEXT_CONFIGS
└── utils/
    └── metrics.py          # partial_auc_above_tpr (competition metric),
                            # sensitivity/specificity at threshold,
                            # evaluate_binary_classification
```

All modules are designed for import in both local scripts and Jupyter notebooks. The Kaggle deployment scripts (`kaggle_notebooks/`) contain self-contained copies of critical classes for environments where `src/` imports are unavailable.
