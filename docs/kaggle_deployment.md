# Kaggle GPU Deployment Guide

This guide explains how to run the complete training pipeline on Kaggle's free GPU.

---

## Quick Start

### 1. Upload Code to Kaggle

Create a new Kaggle notebook and upload these files:

```
Required Files:
├── src/
│   ├── models/
│   │   ├── vit_classifier.py
│   │   └── convnext_classifier.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── augmentations.py
│   ├── features/
│   │   └── patient_features.py
│   └── utils/
│       └── metrics.py
├── scripts/
│   ├── train_vit.py
│   ├── train_convnext.py
│   └── extract_embeddings.py
└── configs/
    ├── eva02_production.yaml
    └── convnext_base.yaml
```

### 2. Setup Kaggle Notebook

```python
# Cell 1: Install dependencies
!pip install timm albumentations segmentation-models-pytorch -q

# Cell 2: Mount data
from kaggle_datasets import KaggleDatasets
DATA_DIR = KaggleDatasets().get_gcp_path('isic-2024-challenge')
```

### 3. Run Training

```python
# Cell 3: Train ViT (takes ~8-10 hours)
!python scripts/train_vit.py \
    --data_dir {DATA_DIR} \
    --model_config eva02_base \
    --output_dir /kaggle/working/outputs/eva02 \
    --epochs 20 \
    --batch_size 24 \
    --accumulation_steps 3 \
    --fp16

# Cell 4: Train ConvNeXt (takes ~6-8 hours)
!python scripts/train_convnext.py \
    --data_dir {DATA_DIR} \
    --model_config convnextv2_nano \
    --output_dir /kaggle/working/outputs/convnext \
    --epochs 20 \
    --batch_size 32 \
    --fp16
```

---

## Optimized Training Strategy

### Option A: Single Session (16-20 hours)

Train both models sequentially in one Kaggle session:

```bash
# Start with smaller model first (in case of timeout)
python scripts/train_convnext.py --config configs/convnext_base.yaml
python scripts/train_vit.py --config configs/eva02_production.yaml
```

**Pros**: Simpler, no state management
**Cons**: Risk of timeout before completion

### Option B: Multi-Session (Recommended)

Split across multiple sessions with checkpointing:

**Session 1** (8-10 hours): Train EVA02
```python
!python scripts/train_vit.py \
    --model_config eva02_base \
    --epochs 20 \
    --fp16
```

**Session 2** (6-8 hours): Train ConvNeXt
```python
!python scripts/train_convnext.py \
    --model_config convnextv2_nano \
    --epochs 20 \
    --fp16
```

**Session 3** (1-2 hours): Extract embeddings & ensemble
```python
# Extract ViT embeddings
!python scripts/extract_embeddings.py \
    --model_dir /kaggle/input/eva02-weights \  # Upload from Session 1
    --model_config eva02_base

# Extract ConvNeXt embeddings
!python scripts/extract_embeddings.py \
    --model_dir /kaggle/input/convnext-weights \  # Upload from Session 2
    --model_config convnextv2_nano

# Train GBDT ensemble (in notebook)
```

### Option C: Fold-by-Fold (Most Flexible)

Train one fold at a time (useful if hitting time limits):

```python
# Session 1: Fold 0
!python scripts/train_vit.py --fold 0 --epochs 20

# Session 2: Fold 1
!python scripts/train_vit.py --fold 1 --epochs 20

# ... continue for folds 2-4
```

---

## Kaggle-Specific Optimizations

### GPU Selection

Kaggle provides different GPUs:
- **P100**: 16GB VRAM - Good for EVA02-Base
- **T4**: 16GB VRAM - Good for all models
- **TPU**: Not recommended (PyTorch support limited)

**Check GPU**:
```python
!nvidia-smi
```

### Memory Management

If running out of memory:

```python
# Reduce batch size
--batch_size 16 --accumulation_steps 4

# Use smaller model
--model_config vit_small  # instead of eva02_base

# Enable gradient checkpointing (saves memory, slower)
# Add to model: model.backbone.set_grad_checkpointing(True)
```

### Saving Outputs

Kaggle notebooks lose `/kaggle/working` after session ends. Save to datasets:

```python
# At end of training
!mkdir -p /kaggle/working/model_weights
!cp -r outputs/* /kaggle/working/model_weights/

# Manually upload /kaggle/working/model_weights as new dataset
# Or use Kaggle API:
!kaggle datasets create -p /kaggle/working/model_weights
```

---

## Training Time Estimates

| Model | Folds | Epochs | Batch Size | GPU | Time |
|-------|-------|--------|------------|-----|------|
| ViT-Tiny | 5 | 20 | 32 | P100 | ~2 hours |
| ViT-Base | 5 | 20 | 28 | P100 | ~6 hours |
| **EVA02-Base** | **5** | **20** | **24** | **P100** | **~10 hours** |
| ConvNeXt-Nano | 5 | 20 | 32 | P100 | ~6 hours |
| ConvNeXt-Tiny | 5 | 20 | 28 | P100 | ~8 hours |

**Total for full ensemble**: ~16-20 hours across 2-3 sessions

---

## Monitoring Training

### Progress Tracking

```python
# Add to training notebook
import matplotlib.pyplot as plt

# After each epoch, log metrics
metrics_df = pd.DataFrame({
    'epoch': epochs,
    'train_loss': train_losses,
    'val_pauc': val_paucs
})

# Plot progress
plt.plot(metrics_df['epoch'], metrics_df['val_pauc'])
plt.title('Validation pAUC Progress')
plt.xlabel('Epoch')
plt.ylabel('pAUC @ 80% TPR')
plt.show()
```

### Expected Metrics

Look for these validation pAUC values per fold:

| Epoch | ViT-Tiny | EVA02-Base | ConvNeXt-Nano |
|-------|----------|------------|---------------|
| 5 | 0.08-0.10 | 0.10-0.12 | 0.09-0.11 |
| 10 | 0.10-0.12 | 0.12-0.14 | 0.11-0.13 |
| 15 | 0.10-0.13 | 0.13-0.15 | 0.12-0.14 |
| **20** | **0.11-0.13** | **0.14-0.16** | **0.13-0.15** |

**Red flags** (investigate if you see):
- pAUC < 0.05 after 5 epochs → Check data loading
- Loss not decreasing → Check learning rate
- Val pAUC decreasing → Overfitting, add regularization

---

## Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

```python
# Solution 1: Reduce batch size
--batch_size 16 --accumulation_steps 4

# Solution 2: Use smaller model
--model_config vit_small

# Solution 3: Reduce image size (not recommended)
# Modify in augmentations.py: image_size=192 instead of 224
```

### Issue 2: Slow Training

```python
# Solution 1: Enable FP16
--fp16

# Solution 2: Reduce num_workers if I/O bottleneck
--num_workers 2

# Solution 3: Use HDF5 instead of JPEG
# Set use_hdf5=True in ISICDataset
```

### Issue 3: Poor Performance

```python
# Check 1: Verify data loading
sample_batch = next(iter(train_loader))
print(sample_batch[0].shape)  # Should be (B, 3, 224, 224)
print(sample_batch[1].unique())  # Should have both 0 and 1

# Check 2: Verify balanced sampling
targets = [batch[1] for batch in islice(train_loader, 10)]
print(np.mean(targets))  # Should be ~0.5

# Check 3: Learning rate
# If val loss plateaus early, increase LR
# If val loss oscillates, decrease LR
```

---

## Final Ensemble on Kaggle

After training all models:

```python
# Load tabular baseline from Phase 2
baseline_preds = np.load('/kaggle/input/baseline-oof/oof_improved.npy')

# Load ViT embeddings
vit_embeddings = np.load('/kaggle/input/eva02-embeddings/train_embeddings.npy')

# Load ConvNeXt embeddings
convnext_embeddings = np.load('/kaggle/input/convnext-embeddings/train_embeddings.npy')

# Load engineered features
features_df = pd.read_parquet('/kaggle/input/engineered-features/train_features.parquet')

# Combine all features
X_combined = np.concatenate([
    features_df[tabular_cols].values,  # ~200 tabular features
    vit_embeddings,                     # 768 ViT features
    convnext_embeddings                 # 640 ConvNeXt features
], axis=1)  # Total: ~1600 features

# Train final GBDT
import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'scale_pos_weight': imbalance_ratio
}

# Train with 5-fold CV
oof_preds = np.zeros(len(X_combined))
test_preds = np.zeros(len(X_test))

for fold in range(5):
    train_idx = df_train['fold'] != fold
    val_idx = df_train['fold'] == fold

    model = lgb.train(
        lgb_params,
        lgb.Dataset(X_combined[train_idx], y_train[train_idx]),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(X_combined[val_idx], y_train[val_idx])],
        callbacks=[lgb.early_stopping(50)]
    )

    oof_preds[val_idx] = model.predict(X_combined[val_idx])
    test_preds += model.predict(X_test) / 5

# Evaluate
from src.utils.metrics import partial_auc_above_tpr
final_pauc = partial_auc_above_tpr(y_train, oof_preds, min_tpr=0.8)
print(f"Final Ensemble pAUC@80TPR: {final_pauc:.4f}")

# Create submission
submission = pd.DataFrame({
    'isic_id': test_ids,
    'target': test_preds
})
submission.to_csv('submission.csv', index=False)
```

---

## Checklist Before Training

- [ ] Uploaded all source files to Kaggle
- [ ] Installed dependencies (`timm`, `albumentations`)
- [ ] Verified data path is correct
- [ ] Set GPU accelerator in notebook settings
- [ ] Configured appropriate batch size for GPU memory
- [ ] Enabled FP16 for faster training
- [ ] Set up checkpointing to save outputs

## Checklist After Training

- [ ] Downloaded model checkpoints
- [ ] Extracted embeddings for ensemble
- [ ] Saved OOF predictions
- [ ] Uploaded trained weights as Kaggle dataset
- [ ] Documented training logs and metrics
- [ ] Ready for final ensemble

---

## Expected Final Performance

| Component | Individual pAUC | Notes |
|-----------|----------------|-------|
| Tabular (engineered) | 0.10-0.12 | Phase 2 baseline |
| EVA02-Base | 0.14-0.16 | Strong image model |
| ConvNeXt-Nano | 0.13-0.15 | Architectural diversity |
| **Multi-modal Ensemble** | **0.16-0.18** | **Competitive** ⭐ |

**Top 3 Kaggle solutions**: 0.17-0.18 pAUC
**Our target**: 0.16-0.18 pAUC (achievable)

---

Good luck with training! 🚀
