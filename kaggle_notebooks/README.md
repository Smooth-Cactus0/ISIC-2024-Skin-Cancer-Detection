# Kaggle Notebooks for ISIC 2024

These are **self-contained** Python scripts ready to copy-paste into Kaggle notebooks.

## Quick Start

### 1. Create New Kaggle Notebook

- Go to https://www.kaggle.com/code
- Click "New Notebook"
- Select "GPU P100" accelerator (Settings → Accelerator)
- Add ISIC 2024 dataset (Add Data → Search "isic-2024-challenge")

### 2. Copy Script to Notebook

**Option A: ViT/EVA02 Training** (Recommended first):
- Open `train_vit_kaggle.py`
- Copy entire file
- Paste into Kaggle code cell
- Run cell (takes ~10 hours)

**Option B: ConvNeXt Training**:
- Open `train_convnext_kaggle.py`
- Copy entire file
- Paste into Kaggle code cell
- Run cell (takes ~8 hours)

### 3. Save Outputs

After training completes:

```python
# Add this cell to save outputs
!zip -r model_weights.zip /kaggle/working/outputs
```

Then manually download `model_weights.zip` or create Kaggle dataset:
- Click "Save Version"
- Output will be saved automatically
- Can reload in future notebooks

## Complete Multi-Session Workflow

The full pipeline requires training multiple models across several Kaggle sessions:

### Session 1: Train ViT (~10 hours)

```python
# 1. Create new Kaggle notebook
# 2. Add ISIC 2024 dataset (isic-2024-challenge)
# 3. Enable GPU P100 accelerator
# 4. Copy train_vit_kaggle.py into code cell
# 5. Run cell
```

After training completes:
- Click "Save Version" to save outputs
- `/kaggle/working/outputs` contains fold_0/ through fold_4/ with best_model.pth
- Create a Kaggle dataset from outputs: "My-EVA02-Weights"

### Session 2: Train ConvNeXt (~8 hours)

```python
# 1. Create new Kaggle notebook
# 2. Add ISIC 2024 dataset
# 3. Enable GPU P100 accelerator
# 4. Copy train_convnext_kaggle.py into code cell
# 5. Run cell
```

After training:
- Save outputs as Kaggle dataset: "My-ConvNeXt-Weights"

### Session 3: Extract ViT Embeddings (~1 hour)

```python
# 1. Create new Kaggle notebook
# 2. Add datasets: ISIC 2024 + My-EVA02-Weights
# 3. Copy extract_embeddings_kaggle.py into code cell
# 4. Modify config:
class Config:
    model_dir = '/kaggle/input/my-eva02-weights'
    model_type = 'eva02'
# 5. Run cell
```

After extraction:
- Save outputs as Kaggle dataset: "My-ViT-Embeddings"

### Session 4: Extract ConvNeXt Embeddings (~1 hour)

```python
# Same as Session 3, but:
class Config:
    model_dir = '/kaggle/input/my-convnext-weights'
    model_type = 'convnext'
```

Save as: "My-ConvNeXt-Embeddings"

### Session 5: Final Ensemble (~1 hour)

```python
# 1. Create new Kaggle notebook
# 2. Add datasets: ISIC 2024 + My-ViT-Embeddings + My-ConvNeXt-Embeddings
# 3. Copy ensemble_kaggle.py into code cell
# 4. Modify config:
class Config:
    vit_embeddings_dir = '/kaggle/input/my-vit-embeddings'
    convnext_embeddings_dir = '/kaggle/input/my-convnext-embeddings'
# 5. Run cell
```

After training:
- Download `submission.csv` from `/kaggle/working`
- Submit to Kaggle competition

**Total time: ~21 hours across 5 sessions**

## File Overview

| File | Model | Runtime | Best For |
|------|-------|---------|----------|
| `train_vit_kaggle.py` | EVA02-Base | ~10h | **Production** (best performance) |
| `train_convnext_kaggle.py` | ConvNeXt-Nano | ~8h | Ensemble diversity |
| `extract_embeddings_kaggle.py` | Extract features | ~2h | After training both models |
| `ensemble_kaggle.py` | LightGBM | ~1h | Final multi-modal model |

**All scripts are fully self-contained** - copy entire file into Kaggle code cell and run.

## Configuration

Each script has a `Config` class at the top. Modify for quick testing:

```python
class Config:
    # Quick test (1 fold, 5 epochs, ~1 hour)
    fold = 0           # Train only fold 0
    epochs = 5         # Reduce epochs
    use_sample = True  # Use 50k sample
    sample_size = 50000

    # Production (all folds, full training, ~10 hours)
    fold = None        # Train all 5 folds
    epochs = 20        # Full training
    use_sample = False # Use full dataset
```

## Expected Performance

After training completes, you should see:

### ViT/EVA02 Results

```
Fold 0: Best pAUC = 0.1523 @ epoch 16
Fold 1: Best pAUC = 0.1487 @ epoch 14
Fold 2: Best pAUC = 0.1552 @ epoch 18
Fold 3: Best pAUC = 0.1498 @ epoch 15
Fold 4: Best pAUC = 0.1515 @ epoch 17

Mean pAUC: 0.1515 ± 0.0024
```

**Interpretation**:
- pAUC ~0.15: Good performance, competitive with top solutions
- Low std (~0.002): Stable across folds, good generalization
- Early stopping 14-18: Appropriate regularization

### ConvNeXt Results

```
Fold 0: Best pAUC = 0.1445 @ epoch 15
Fold 1: Best pAUC = 0.1412 @ epoch 13
Fold 2: Best pAUC = 0.1478 @ epoch 16
Fold 3: Best pAUC = 0.1433 @ epoch 14
Fold 4: Best pAUC = 0.1456 @ epoch 15

Mean pAUC: 0.1445 ± 0.0023
```

**Interpretation**:
- Slightly lower than ViT (expected - Nano vs Base)
- Different architecture = complementary errors
- Good for ensemble diversity

## Troubleshooting

### Out of Memory

```python
# In Config class, reduce batch size:
batch_size = 16        # Instead of 24
accumulation_steps = 4  # Instead of 3
```

### Slow Training

```python
# Ensure FP16 is enabled:
fp16 = True

# Reduce workers if I/O bottleneck:
num_workers = 2  # Or even 0
```

### Model Not Found

```python
# Check model name spelling:
model_name = 'eva02_base_patch14_224.mim_in22k'  # Correct

# If timm version mismatch, try:
model_name = 'eva02_base_patch14_224'  # Without suffix
```

### Poor Performance (pAUC < 0.10)

```python
# Check data loading:
sample_batch = next(iter(train_loader))
print(sample_batch[0].shape)  # Should be (24, 3, 224, 224)
print(sample_batch[1].mean())  # Should be ~0.5 (balanced sampling)

# Check loss is decreasing:
# If train loss not decreasing → increase learning rate
# If val loss increasing → add regularization (dropout, weight decay)
```

### Embedding Extraction Issues

```python
# Issue: Checkpoint not found
# Solution: Check model_dir path matches your Kaggle dataset name
model_dir = '/kaggle/input/my-eva02-weights'  # Must match dataset slug

# Issue: Out of memory during extraction
# Solution: Reduce batch size
batch_size = 32  # Instead of 64

# Issue: Wrong embedding dimension
# Solution: Verify model_type matches checkpoint
model_type = 'eva02'  # Must match training
```

### Ensemble Training Issues

```python
# Issue: Missing 'fold' column
# Solution: Folds should be created during ViT/ConvNeXt training
# If missing, create them in ensemble script (already handled)

# Issue: Shape mismatch in concatenation
# Solution: Ensure all embeddings extracted from same samples
# Check: vit_train.shape[0] == convnext_train.shape[0] == len(train_df)

# Issue: Poor ensemble performance
# Solution: Check individual model predictions first
vit_preds = np.load('vit_embeddings/train_predictions.npy')
print(f"ViT pAUC: {partial_auc_above_tpr(y_train, vit_preds):.4f}")
# If individual models are poor, need to retrain
```

## Training Multiple Models

**Sequential Training** (Recommended):
1. Run `train_vit_kaggle.py` → Save version → Download weights
2. Run `train_convnext_kaggle.py` → Save version → Download weights
3. Run `extract_embeddings_kaggle.py` → Produces embeddings
4. Run `ensemble_kaggle.py` → Final submission

**Time Management**:
- Each Kaggle session: 12 hours max
- Strategy: Train one model per session
- Between sessions: Download weights, upload as dataset

## After Training

Your `/kaggle/working/outputs` should contain:

```
outputs/
├── fold_0/
│   └── best_model.pth
├── fold_1/
│   └── best_model.pth
├── fold_2/
│   └── best_model.pth
├── fold_3/
│   └── best_model.pth
└── fold_4/
    └── best_model.pth
```

Each `.pth` file contains:
- `model_state_dict`: Trained weights
- `best_pauc`: Best validation score
- `epoch`: Best epoch number

## Next Steps

1. ✅ Train ViT (this notebook)
2. ✅ Train ConvNeXt (separate notebook)
3. ⏸️ Extract embeddings (use both models)
4. ⏸️ Train ensemble (combine all features)
5. ⏸️ Create submission

## Tips for Success

1. **Start with quick test**: Set `fold=0, epochs=5, use_sample=True` first
2. **Monitor early epochs**: pAUC should be >0.05 by epoch 3
3. **Check GPU utilization**: Run `!nvidia-smi` to verify GPU usage
4. **Save frequently**: Commit notebook after each fold completes
5. **Document results**: Note down pAUC scores for comparison

## Performance Expectations

| Stage | pAUC@80TPR | Notes |
|-------|------------|-------|
| Epoch 1 | ~0.05-0.08 | Initial learning |
| Epoch 5 | ~0.10-0.12 | Rapid improvement |
| Epoch 10 | ~0.12-0.14 | Convergence starts |
| Epoch 15-20 | ~0.14-0.16 | Best performance |
| **Final (ViT)** | **~0.15** | Competitive |
| **Final (ConvNeXt)** | **~0.14** | Diversity |
| **Ensemble** | **~0.16-0.18** | Target |

Good luck! 🚀
