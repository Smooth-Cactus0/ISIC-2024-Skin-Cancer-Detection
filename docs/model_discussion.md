# Model Architecture Discussion

## Vision Transformer: EVA02

### Architecture Overview

EVA02 (Exploring the limits of Visual representation learning at scAle) is a Vision Transformer variant that achieves state-of-the-art performance through a combination of:

1. **Masked Image Modeling (MIM) pretraining**: The model learns visual representations by predicting masked image patches, similar to BERT's masked language modeling. This self-supervised pretraining on ImageNet-21K produces representations that capture fine-grained texture and structure information.

2. **CLIP-guided distillation**: EVA02 uses a CLIP vision encoder as a teacher during MIM pretraining, aligning its representations with text-image understanding. This produces features that encode semantic meaning (e.g., "irregular border" → potentially malignant).

3. **Rotary Position Embedding (RoPE)**: Unlike standard absolute position embeddings, RoPE encodes relative spatial relationships, enabling the model to better capture spatial patterns regardless of position in the image.

### Why EVA02 for Dermatology?

| Property | Benefit for Skin Cancer Detection |
|----------|-----------------------------------|
| **ImageNet-21K pretraining** | Rich texture vocabulary (important for skin patterns) |
| **Self-supervised MIM** | Learns to reconstruct fine details like border irregularity |
| **Global attention** | Captures whole-lesion symmetry patterns |
| **768-d embeddings** | Rich feature space for downstream GBDT |

### Model Configurations Used

```python
MODEL_CONFIGS = {
    'eva02_small': {
        'model_name': 'eva02_small_patch14_336.mim_in22k_ft_in1k',
        'feature_dim': 384,
        'img_size': 336
    },
    'eva02_base': {
        'model_name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
        'feature_dim': 768,
        'img_size': 448
    }
}
```

The **base** model (768-d) is used for production; the **small** model (384-d) for rapid iteration and debugging.

---

## ConvNeXt V2

### Architecture Overview

ConvNeXtV2 is a "modernized ResNet" that matches or exceeds ViT performance while maintaining the inductive biases of convolutional architectures:

1. **Hierarchical feature maps**: Unlike ViT's single-resolution processing, ConvNeXt produces multi-scale features — critical for capturing both fine (texture) and coarse (shape) patterns.

2. **Fully Convolutional Masked Autoencoder (FCMAE)**: V2's pretraining applies masked autoencoding to convolutional networks, improving feature quality over supervised-only pretraining.

3. **Global Response Normalization (GRN)**: A new normalization layer that enhances feature diversity by normalizing across channels, reducing redundancy in learned features.

### Why ConvNeXt for Ensemble Diversity?

The key insight is **architectural complementarity**:

```
ViT (EVA02):     [Image] → [Patches] → [Global Self-Attention] → [Single-scale features]
                  Strong at: global structure, symmetry, overall shape

ConvNeXt V2:     [Image] → [Conv layers] → [Hierarchical features] → [Multi-scale features]
                  Strong at: local textures, border gradients, fine details
```

In melanoma detection, both global (asymmetry, overall color distribution) and local (border irregularity, color hotspots) features are diagnostic. An ensemble captures both.

### Stochastic Depth

ConvNeXt uses **stochastic depth** (drop_path) for regularization:

```python
# During training, randomly skip layers with probability p
# This acts as implicit ensemble of sub-networks
drop_path_rate = 0.1  # 10% of layers randomly dropped per forward pass
```

With only 393 positive training samples, regularization is critical to prevent overfitting.

---

## Focal Loss

### The Problem with Standard Cross-Entropy

With 1000:1 class imbalance, standard binary cross-entropy is dominated by the loss from the 400,000 benign samples. Even if the model perfectly classifies all malignant samples, benign misclassifications generate overwhelming gradient signal.

### Focal Loss Formulation

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

where:
  p_t = model's predicted probability for the correct class
  α_t = class weight (α for positive, 1-α for negative)
  γ   = focusing parameter (higher → more focus on hard examples)
```

**Key parameters**:
- **α = 0.25**: Gives positive samples 4× weight (but less than 1000× — focal modulation handles the rest)
- **γ = 2.0**: Standard focusing factor — an "easy" sample with p_t=0.9 gets (1-0.9)² = 0.01× the loss of a sample at chance level

### Effect on Training Dynamics

```
Example (benign sample, correctly classified with p=0.95):
  CE loss:    -log(0.95)                    = 0.051
  Focal loss: -0.75 × (1-0.95)² × log(0.95) = 0.75 × 0.0025 × 0.051 = 0.000096
                                                          ↑
                                              96% reduction in loss for easy samples
```

This frees the model's capacity to focus on the ~393 malignant samples and the hardest benign samples (those bordering the decision boundary).

### Implementation

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)  # p_t = sigmoid(input) for correct class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()
```

---

## Patient Normalization

### Motivation

Consider two patients:
- **Patient A**: Fair skin, all lesions have low melanin → absolute color values are low
- **Patient B**: Dark skin, all lesions have high melanin → absolute color values are high

A malignant lesion on Patient A may have absolute color values *lower* than Patient B's benign lesions. Absolute features are confounded by patient-level factors.

### Solution: Relative Features

For each lesion, compute how it differs from the patient's *other* lesions:

```python
# Z-score: How many standard deviations from the patient mean?
z_score = (lesion_value - patient_mean) / patient_std

# Percentile: Where does this lesion rank among the patient's lesions?
percentile = rank_within_patient / n_patient_lesions

# Deviation: How far from the patient median?
deviation = lesion_value - patient_median

# Flags: Is this the most extreme lesion?
is_max = (lesion_value == patient_max)
is_min = (lesion_value == patient_min)
```

### Clinical Parallel

This directly mirrors dermatological practice: during a total body examination, dermatologists look for the "ugly duckling" — the lesion that stands out from a patient's other moles. Patient normalization is the computational equivalent of the ugly duckling sign.

### Statistical Impact

Patient-normalized features consistently improve GBDT performance:

| Feature Set | Expected CV pAUC |
|-------------|------------------|
| Raw TBP features only | 0.10–0.12 |
| + Patient-normalized | 0.12–0.14 |
| + ABCDE clinical | 0.13–0.15 |

The ~2 point improvement from patient normalization is significant — it's the single most impactful feature engineering technique.

---

## Ensemble Strategy

### Why Ensemble?

No single model captures all aspects of malignancy:
- **ViT** excels at global pattern recognition (symmetry, overall shape)
- **ConvNeXt** captures local features (border detail, texture gradient)
- **GBDT** handles tabular metadata features (patient history, demographics)

### Rank Averaging

Rather than averaging raw probabilities (which can be dominated by poorly calibrated models), we use **rank averaging**:

```python
# Convert predictions to ranks (0 to 1)
for model_name in models:
    preds[model_name] = rankdata(preds[model_name]) / len(preds[model_name])

# Average ranks
final_pred = sum(preds.values()) / len(preds)
```

Rank averaging is:
1. **Scale-invariant**: Works regardless of each model's calibration
2. **Robust to outliers**: A single model's extreme prediction can't dominate
3. **Simple**: No hyperparameters to tune

### Expected Contribution

| Component | Standalone pAUC | Contribution to Ensemble |
|-----------|----------------|--------------------------|
| EVA02 embeddings + GBDT | 0.14–0.16 | Primary signal |
| ConvNeXt embeddings + GBDT | 0.13–0.15 | Diversity boost (+0.01) |
| Tabular features + GBDT | 0.12–0.14 | Complementary signal |
| **Rank-averaged ensemble** | **0.16–0.18** | **Best combined** |

---

## Computational Considerations

### Training Time Budget (Kaggle Free GPU)

| Component | GPU Hours | Sessions |
|-----------|-----------|----------|
| EVA02 5-fold | ~10h | 1 session |
| ConvNeXt 5-fold | ~8h | 1 session |
| Embedding extraction | ~2h | 1 session |
| GBDT ensemble | ~1h | 1 session |
| **Total** | **~21h** | **4 sessions** |

Kaggle provides ~30h/week of free GPU time (P100), so the full pipeline is trainable within a single week.

### Mixed Precision Training

Using FP16 via PyTorch's `torch.cuda.amp`:
- **Memory**: ~40% reduction in GPU memory usage
- **Speed**: ~2× faster on P100 GPU
- **Accuracy**: Negligible impact with proper loss scaling

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Inference Optimization

For Kaggle submission time limits:
- **Batch inference**: Process all test images in batches of 32
- **No TTA on test**: Skip test-time augmentation for speed (marginal gain)
- **Single model per architecture**: Use best fold model instead of all 5 folds
