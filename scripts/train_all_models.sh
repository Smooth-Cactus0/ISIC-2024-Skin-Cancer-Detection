#!/bin/bash
# Comprehensive training script for all models
# Run this on Kaggle GPU to train the complete ensemble

set -e  # Exit on error

echo "================================================================================"
echo "ISIC 2024 - Complete Model Training Pipeline"
echo "================================================================================"

# Configuration
DATA_DIR="../isic-2024-challenge"
EPOCHS=20
SEED=42

# Phase 1: Vision Transformer (ViT-Tiny for quick validation)
echo ""
echo "Phase 1: Training ViT-Tiny (Validation)"
echo "================================================================================"
python train_vit.py \
    --data_dir $DATA_DIR \
    --model_config vit_tiny \
    --output_dir ../outputs/vit_tiny \
    --epochs $EPOCHS \
    --batch_size 32 \
    --fp16 \
    --seed $SEED

# Phase 2: Vision Transformer (EVA02-Base for production)
echo ""
echo "Phase 2: Training EVA02-Base (Production)"
echo "================================================================================"
python train_vit.py \
    --data_dir $DATA_DIR \
    --model_config eva02_base \
    --output_dir ../outputs/eva02_base \
    --epochs 30 \
    --batch_size 24 \
    --accumulation_steps 3 \
    --fp16 \
    --seed $SEED

# Phase 3: ConvNeXt (Nano - from 3rd place solution)
echo ""
echo "Phase 3: Training ConvNeXtV2-Nano"
echo "================================================================================"
python train_convnext.py \
    --data_dir $DATA_DIR \
    --model_config convnextv2_nano \
    --output_dir ../outputs/convnext_nano \
    --epochs $EPOCHS \
    --batch_size 32 \
    --fp16 \
    --seed $SEED

# Phase 4: ConvNeXt (Tiny for ensemble diversity)
echo ""
echo "Phase 4: Training ConvNeXtV2-Tiny"
echo "================================================================================"
python train_convnext.py \
    --data_dir $DATA_DIR \
    --model_config convnextv2_tiny \
    --output_dir ../outputs/convnext_tiny \
    --epochs $EPOCHS \
    --batch_size 28 \
    --accumulation_steps 2 \
    --fp16 \
    --seed $SEED

# Phase 5: Extract embeddings from ViT models
echo ""
echo "Phase 5: Extracting ViT Embeddings"
echo "================================================================================"
python extract_embeddings.py \
    --data_dir $DATA_DIR \
    --model_dir ../outputs/eva02_base \
    --model_config eva02_base \
    --output_dir ../outputs/embeddings/eva02

# Phase 6: Extract embeddings from ConvNeXt models
echo ""
echo "Phase 6: Extracting ConvNeXt Embeddings"
echo "================================================================================"
python extract_embeddings.py \
    --data_dir $DATA_DIR \
    --model_dir ../outputs/convnext_nano \
    --model_config convnextv2_nano \
    --output_dir ../outputs/embeddings/convnext

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  ✓ ViT-Tiny trained (validation)"
echo "  ✓ EVA02-Base trained (production)"
echo "  ✓ ConvNeXtV2-Nano trained (3rd place architecture)"
echo "  ✓ ConvNeXtV2-Tiny trained (diversity)"
echo "  ✓ Embeddings extracted"
echo ""
echo "Next Steps:"
echo "  1. Run notebooks/03_ensemble.ipynb to combine models"
echo "  2. Train final GBDT on combined features"
echo "  3. Generate submission file"
echo ""
echo "Expected Performance:"
echo "  - Individual models: pAUC ~0.14-0.16"
echo "  - Multi-modal ensemble: pAUC ~0.16-0.18"
echo ""
