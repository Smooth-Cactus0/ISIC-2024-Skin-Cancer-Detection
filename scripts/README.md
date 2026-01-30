# Training Scripts

This directory contains training and inference scripts for the ISIC 2024 models.

## Quick Start

### 1. Train Vision Transformer (Tiny - for testing)

```bash
# Quick test with tiny model
python train_vit.py \
    --model_config vit_tiny \
    --epochs 5 \
    --batch_size 32 \
    --fold 0 \
    --use_sample

# Expected time: ~10 minutes on GPU
# Expected pAUC: ~0.10-0.12 (tiny model, 1 fold)
```

### 2. Train EVA02 (Production - full pipeline)

```bash
# Full training with EVA02 Base
python train_vit.py \
    --model_config eva02_base \
    --epochs 20 \
    --batch_size 24 \
    --accumulation_steps 3 \
    --fp16

# Expected time: ~8-12 hours on single GPU
# Expected pAUC: ~0.14-0.16 (competitive)
```

### 3. Extract Embeddings for Ensemble

```bash
# After training, extract embeddings
python extract_embeddings.py \
    --model_dir ../outputs/vit_tiny \
    --model_config vit_tiny \
    --output_dir ../outputs/embeddings

# Creates train/test embeddings for GBDT ensemble
```

## Model Configurations

Available via `--model_config`:

| Config | Model | Params | Feature Dim | Speed | Performance |
|--------|-------|--------|-------------|-------|-------------|
| `vit_tiny` | ViT-Tiny | 5M | 192 | Very Fast | Baseline |
| `vit_small` | ViT-Small | 22M | 384 | Fast | Good |
| `vit_base` | ViT-Base | 86M | 768 | Medium | Very Good |
| `eva02_small` | EVA02-Small | 22M | 384 | Fast | Very Good |
| `eva02_base` | EVA02-Base | 86M | 768 | Medium | **Best** ⭐ |

## Configuration Files

Instead of command-line arguments, use YAML configs:

```bash
# Using config file
python train_vit.py --config ../configs/vit_base.yaml

# Override specific parameters
python train_vit.py --config ../configs/eva02_production.yaml --epochs 10
```

## Key Training Parameters

### Learning Rates (Critical)

```bash
--lr 1e-3              # Head learning rate (higher for new layers)
--lr_backbone 1e-5     # Backbone LR (lower for pretrained weights)
```

**Why differential LR?** Pretrained backbone already has good features; we fine-tune gently while learning head from scratch.

### Focal Loss (For Imbalance)

```bash
--focal_alpha 0.25     # Weight for positive class
--focal_gamma 2.0      # Focusing parameter (higher = more focus on hard examples)
```

**Why Focal Loss?** With 1000:1 imbalance, standard BCE is dominated by easy negatives. Focal loss down-weights these.

### Mixed Precision (Speed)

```bash
--fp16                 # Enable FP16 training
```

**Speedup**: 2-3x faster with minimal accuracy impact. Recommended for production.

### Freezing Strategy

```bash
--freeze_epochs 3      # Freeze backbone for first 3 epochs
```

**Why freeze?** Prevents catastrophic forgetting of pretrained features while head learns basic discrimination.

## Output Structure

After training, outputs are organized as:

```
outputs/
└── vit_tiny/          # or eva02_base, etc.
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

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `best_pauc`: Best validation pAUC@80TPR
- `epoch`: Best epoch number
- `config`: Model configuration
- `args`: Training arguments

## Embedding Extraction

Embeddings are extracted from the **backbone** (before classification head):

```
Input Image (224x224x3)
    ↓
EVA02 Backbone
    ↓
Feature Vector (768-dim)  ← Extract this
    ↓
Classification Head
    ↓
Logit (1-dim)
```

These embeddings are then used as features for GBDT ensemble:

```python
# Final ensemble features
tabular_features = [age, sex, tbp_lv_*, engineered_features]  # ~200 features
image_features = [vit_emb_0, vit_emb_1, ..., vit_emb_767]     # 768 features
combined = tabular_features + image_features                    # ~968 features

# Train LightGBM/XGBoost on combined features
gbdt_model.fit(combined, target)
```

## Performance Expectations

Based on top Kaggle solutions and our setup:

| Approach | Expected pAUC@80TPR | Notes |
|----------|---------------------|-------|
| Tabular baseline (raw) | ~0.08-0.10 | From Phase 2 |
| Tabular + engineered | ~0.10-0.12 | From Phase 2 |
| ViT-Tiny (image only) | ~0.10-0.12 | Quick baseline |
| ViT-Base (image only) | ~0.12-0.14 | Good performance |
| EVA02-Base (image only) | ~0.14-0.16 | Competitive |
| **Multi-modal ensemble** | **~0.16-0.18** | **Target** ⭐ |

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 16 --accumulation_steps 4

# Use smaller model
--model_config vit_small
```

### Slow Training

```bash
# Enable mixed precision
--fp16

# Reduce workers if CPU bottleneck
--num_workers 2

# Train single fold first
--fold 0
```

### Poor Performance

```bash
# Check data loading
--use_sample  # Test on 50k samples first

# Adjust learning rate
--lr 5e-4 --lr_backbone 1e-5

# Increase augmentation
--augmentation heavy
```

## Next Steps

After Phase 3 (ViT training):

1. **Phase 4**: ConvNeXt model (architectural diversity)
2. **Phase 5**: Multi-task model (segmentation + classification)
3. **Phase 6**: Multi-modal GBDT ensemble
4. **Phase 7**: Final polish and documentation
